#!/usr/bin/env python3
"""
Unified Evaluation Script for BiasedCodebookExp_v2

Supports three execution phases for SLURM parallelization:
  --phase val-search   : (GPU, 4 tasks)  Collect data + search best params -> save params JSON
  --phase test-apply   : (GPU, 24 tasks) Load params + collect & eval ONE test dataset
  --phase aggregate    : (CPU, 4 tasks)  Combine per-dataset results into summary
  --phase all          : (GPU, 4 tasks)  Run everything sequentially

Tuning Strategy:
  ID params:  tuned on own VAL only                           -> best_params_id.json
  OOD params: tuned on own VAL+TEST + other 3 IDs' TEST      -> best_params_ood.json

Methods (9):
  Base:
    - Baseline:      E2V head on raw features (no codebook)
    - Bal_L32:       E2V head on balanced reconstruction at fixed layer 32
    - Bal_Select:    E2V head on balanced reconstruction at best layer (tuned)

  Biased (argmax cosine among biased codebooks):
    - Unfilt:        layer tuned via cross-val
    - Unfilt_ID:     layer tuned on ID TEST only

  Biased_Filt variants (argmax delta-cosine with different fallbacks):
    - Filt:          layer tuned via cross-val;     fallback -> baseline
    - Filt_ID:       layer tuned on ID TEST only;   fallback -> baseline
    - Filt_FB32:     layer tuned via cross-val;     fallback -> balanced L32 -> E2V
    - Filt_FBsel:    layer tuned via cross-val;     fallback -> balanced best-layer (CV) -> E2V

Test scope: EN-only (4 ID + msp + 4 CAMEO EN = 6 per source)
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.dataset_config import DATASET_CONFIGS
from core.config import (
    E2V_LABELS, E2V_HEAD_PATH, DEFAULT_LAYERS,
    FAIR_EMOTIONS, FAIR_E2V_INDICES, DATASET_TO_FAIR_MAP,
    CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR, CAMEO_DATASETS,
    set_seed,
)
from core.quantize import load_codebook, compute_similarity, get_all_reconstructions
from core.classify import E2VClassificationHead, load_e2v_head, load_custom_head
from core.features import get_emotion2vec_extractor, extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']

EN_CAMEO = ['cameo_emns', 'cameo_enterface', 'cameo_jl_corpus']
EN_STANDALONE = ['savee', 'tess', 'meld', 'asvp_esd']

EN_OOD_DATASETS = ['msp'] + EN_CAMEO + EN_STANDALONE
ALL_OOD_DATASETS = EN_OOD_DATASETS

def get_ood_datasets(cb_source: str) -> list:
    """OOD list per source (fixed, same for all sources)."""
    return EN_OOD_DATASETS

BIASED_EMOTIONS = FAIR_EMOTIONS  # ['angry', 'happy', 'neutral', 'sad']


# ==================================================================
# Phase 1: Collect sample-level data
# ==================================================================

def collect_sample_data(
    codebook_source: str,
    test_datasets: List[str],
    balanced_model,
    biased_models: Dict[str, object],
    e2v_head: E2VClassificationHead,
    extractor,
    layers: List[int],
    max_samples: int,
    device: str,
    split: str = 'test',
    metric: str = 'cosine',
) -> Dict[str, List[dict]]:
    """Collect per-sample metrics for datasets.
    
    Args:
        split: 'val' for parameter selection on ID,
               'test' for ID final evaluation,
               'val+test' to merge both splits (used for OOD evaluation).
        metric: 'cosine' or 'l2'. Similarity score stored such that higher = more similar.
    """
    all_data = {}

    for test_ds in test_datasets:
        fair_map = DATASET_TO_FAIR_MAP.get(test_ds)
        if not fair_map:
            logger.warning(f"No fair4 mapping for {test_ds}, skipping")
            continue

        splits_dir = SPLITS_DIR / test_ds

        if split == 'val+test':
            splits_to_load = ['val', 'test']
        else:
            splits_to_load = [split]

        merged_files = defaultdict(list)
        for sp in splits_to_load:
            sp_file = splits_dir / f'{sp}.json'
            if not sp_file.exists():
                if sp == 'val':
                    logger.info(f"  {test_ds}: no {sp}.json, using test only")
                else:
                    logger.warning(f"  {test_ds}: split file not found: {sp_file}")
                continue
            with open(sp_file) as f:
                sp_data = json.load(f)
            for emo, files in sp_data.items():
                merged_files[emo].extend(files)

        if not merged_files:
            continue

        samples = []
        for orig_emo, files in merged_files.items():
            fair_emo = fair_map.get(orig_emo)
            if fair_emo and fair_emo in FAIR_EMOTIONS:
                for fpath in files[:max_samples]:
                    samples.append((fpath, fair_emo))

        if not samples:
            continue

        logger.info(f"  {test_ds}: {len(samples)} samples")
        ds_records = []

        for audio_path, true_label in tqdm(samples, desc=test_ds, leave=False):
            features = extract_features(extractor, audio_path)
            if features is None:
                continue

            features_gpu = features.to(device)
            record = {'true_label': true_label}

            # Baseline classification
            with torch.no_grad():
                logits = e2v_head(features_gpu.unsqueeze(0) if features_gpu.dim() == 2 else features_gpu)
                valid_logits = logits[:, FAIR_E2V_INDICES]
                probs = F.softmax(valid_logits, dim=-1).squeeze(0).cpu().numpy()

            bl_softmax = {emo: float(probs[i]) for i, emo in enumerate(FAIR_EMOTIONS)}
            bl_pred = FAIR_EMOTIONS[int(np.argmax(probs))]
            record['baseline'] = {
                'softmax': bl_softmax,
                'prediction': bl_pred,
                'correct': bl_pred == true_label,
            }

            # Balanced reconstruction cosines per layer
            if balanced_model is not None:
                bal_recons = get_all_reconstructions(balanced_model, features_gpu, layers, device)
                bal_data = {}
                for layer in layers:
                    if layer in bal_recons:
                        sim = compute_similarity(features_gpu, bal_recons[layer], metric)
                        with torch.no_grad():
                            r_logits = e2v_head(bal_recons[layer].unsqueeze(0) if bal_recons[layer].dim() == 2 else bal_recons[layer])
                            r_probs = F.softmax(r_logits[:, FAIR_E2V_INDICES], dim=-1).squeeze(0).cpu().numpy()
                        bal_pred = FAIR_EMOTIONS[int(np.argmax(r_probs))]
                        bal_data[str(layer)] = {'cosine': sim, 'prediction': bal_pred}
                record['balanced'] = bal_data

            # Biased reconstruction cosines per layer per emotion
            biased_data = {}
            for emo, model in biased_models.items():
                emo_recons = get_all_reconstructions(model, features_gpu, layers, device)
                emo_data = {}
                for layer in layers:
                    if layer in emo_recons:
                        sim = compute_similarity(features_gpu, emo_recons[layer], metric)
                        with torch.no_grad():
                            r_logits = e2v_head(emo_recons[layer].unsqueeze(0) if emo_recons[layer].dim() == 2 else emo_recons[layer])
                            r_probs = F.softmax(r_logits[:, FAIR_E2V_INDICES], dim=-1).squeeze(0).cpu().numpy()
                        cls_pred = FAIR_EMOTIONS[int(np.argmax(r_probs))]
                        emo_data[str(layer)] = {
                            'cosine': sim,
                            'cls_prediction': cls_pred,
                        }
                biased_data[emo] = emo_data
            record['biased'] = biased_data

            ds_records.append(record)

        all_data[test_ds] = ds_records
        logger.info(f"  {test_ds}: collected {len(ds_records)} records")

    return all_data


# ==================================================================
# Phase 2: Evaluate methods (vectorized, CPU)
# ==================================================================

def evaluate_basic_methods(records: List[dict], layers: List[int]):
    """Evaluate Baseline, Bal_L32, Bal_Select, Unfilt, Biased_Filt at each layer."""
    N = len(records)
    results = {}

    # Baseline
    bl_correct = sum(1 for r in records if r['baseline']['correct'])
    results['Baseline'] = bl_correct / N

    for layer in layers:
        sl = str(layer)

        # Bal_Select (per-layer, best selected later)
        bal_correct = sum(
            1 for r in records
            if sl in r.get('balanced', {}) and r['balanced'][sl]['prediction'] == r['true_label']
        )
        results[f'Bal_Select-L{layer}'] = bal_correct / N

        # Unfilt: argmax cosine among biased codebooks (no delta filtering)
        unfilt_correct = 0
        for r in records:
            best_emo, best_cos = None, -float('inf')
            for emo, emo_data in r['biased'].items():
                if sl in emo_data and emo_data[sl]['cosine'] > best_cos:
                    best_cos = emo_data[sl]['cosine']
                    best_emo = emo
            if best_emo == r['true_label']:
                unfilt_correct += 1
        results[f'Unfilt-L{layer}'] = unfilt_correct / N

        # Biased_Filt: argmax (biased_cos - balanced_cos), fallback to baseline
        delta_correct = 0
        for r in records:
            bal_cos = r.get('balanced', {}).get(sl, {}).get('cosine', None)
            if bal_cos is None:
                if r['baseline']['correct']:
                    delta_correct += 1
                continue
            best_emo, best_delta = None, -float('inf')
            for emo, emo_data in r['biased'].items():
                if sl in emo_data:
                    delta = emo_data[sl]['cosine'] - bal_cos
                    if delta > best_delta:
                        best_delta = delta
                        best_emo = emo
            if best_delta <= 0 or best_emo is None:
                if r['baseline']['correct']:
                    delta_correct += 1
            elif best_emo == r['true_label']:
                delta_correct += 1
        results[f'Biased_Filt-L{layer}'] = delta_correct / N

    # Bal_L32: fixed layer 32
    results['Bal_L32'] = results.get(f'Bal_Select-L{max(layers)}', 0)

    return results


def _eval_biased_filt(records, layer_str, fallback_fn):
    """Evaluate Biased_Filt with a parameterized fallback function.

    For each sample: if max(biased_cos - balanced_cos) > 0, predict the
    emotion with the highest delta; otherwise call fallback_fn(record) to
    determine if the sample is counted as correct.
    """
    N = len(records)
    if N == 0:
        return 0.0
    correct = 0
    for r in records:
        bal_cos = r.get('balanced', {}).get(layer_str, {}).get('cosine', None)
        if bal_cos is None:
            if fallback_fn(r):
                correct += 1
            continue
        best_emo, best_d = None, -float('inf')
        for emo, ed in r['biased'].items():
            if layer_str in ed:
                d = ed[layer_str]['cosine'] - bal_cos
                if d > best_d:
                    best_d = d
                    best_emo = emo
        if best_d <= 0 or best_emo is None:
            if fallback_fn(r):
                correct += 1
        elif best_emo == r['true_label']:
            correct += 1
    return correct / N


def search_best_params(records: List[dict], layers: List[int]):
    """Search optimal layer for each method via exhaustive single-layer search."""
    best = {}

    basic = evaluate_basic_methods(records, layers)
    best['Baseline'] = {'acc': basic['Baseline'], 'params': {}}

    best['Bal_L32'] = {'acc': basic['Bal_L32'], 'params': {'layer': max(layers)}}
    logger.info(f"  Bal_L32: L{max(layers)} -> {basic['Bal_L32']:.4%}")

    for prefix in ['Bal_Select', 'Unfilt', 'Biased_Filt']:
        best_layer, best_acc = 1, 0
        for layer in layers:
            key = f'{prefix}-L{layer}'
            if key in basic and basic[key] > best_acc:
                best_acc = basic[key]
                best_layer = layer
        best[prefix] = {'acc': best_acc, 'params': {'layer': best_layer}}
        logger.info(f"  {prefix}: L{best_layer} -> {best_acc:.4%}")

    return best


# ==================================================================
# Phase 3: Apply best params to all datasets
# ==================================================================

def apply_params(records: List[dict], layers: List[int], best_params: dict):
    """Apply best parameters to evaluate a dataset.

    Evaluates all 8 methods using the tuned layer parameters:
    - Baseline, Bal_L32, Bal_Select: reference methods
    - Unfilt: argmax biased cosine (no delta filtering)
    - Filt_*: delta-cosine variants with different fallbacks
    """
    N = len(records)
    if N == 0:
        return {}
    results = {}

    # Baseline
    results['Baseline'] = sum(1 for r in records if r['baseline']['correct']) / N

    # Bal_L32 (fixed layer 32)
    sl32 = str(best_params['Bal_L32']['params']['layer'])
    results['Bal_L32'] = sum(
        1 for r in records
        if sl32 in r.get('balanced', {}) and r['balanced'][sl32]['prediction'] == r['true_label']
    ) / N

    # Bal_Select (best layer from tuning)
    bal_sel_layer = str(best_params['Bal_Select']['params']['layer'])
    results['Bal_Select'] = sum(
        1 for r in records
        if bal_sel_layer in r.get('balanced', {})
        and r['balanced'][bal_sel_layer]['prediction'] == r['true_label']
    ) / N

    # Unfilt: argmax cosine among biased codebooks (CV layer)
    unfilt_layer = str(best_params['Unfilt']['params']['layer'])
    correct = 0
    for r in records:
        best_emo, best_cos = None, -float('inf')
        for emo, ed in r['biased'].items():
            if unfilt_layer in ed and ed[unfilt_layer]['cosine'] > best_cos:
                best_cos = ed[unfilt_layer]['cosine']
                best_emo = emo
        if best_emo == r['true_label']:
            correct += 1
    results['Unfilt'] = correct / N

    # Unfilt_ID: argmax cosine among biased codebooks (ID TEST layer)
    unfilt_id_layer = str(best_params['Unfilt_ID']['params']['layer'])
    correct = 0
    for r in records:
        best_emo, best_cos = None, -float('inf')
        for emo, ed in r['biased'].items():
            if unfilt_id_layer in ed and ed[unfilt_id_layer]['cosine'] > best_cos:
                best_cos = ed[unfilt_id_layer]['cosine']
                best_emo = emo
        if best_emo == r['true_label']:
            correct += 1
    results['Unfilt_ID'] = correct / N

    # Filt: layer from Biased_Filt (cross-val), fallback -> baseline
    filt_cv_layer = str(best_params['Biased_Filt']['params']['layer'])
    results['Filt'] = _eval_biased_filt(
        records, filt_cv_layer,
        lambda r: r['baseline']['correct'],
    )

    # Filt_ID: layer from Filt_ID params (ID TEST only), fallback -> baseline
    filt_id_layer = str(best_params['Filt_ID']['params']['layer'])
    results['Filt_ID'] = _eval_biased_filt(
        records, filt_id_layer,
        lambda r: r['baseline']['correct'],
    )

    # Filt_FB32: layer from Biased_Filt (cross-val), fallback -> balanced L32
    results['Filt_FB32'] = _eval_biased_filt(
        records, filt_cv_layer,
        lambda r: (sl32 in r.get('balanced', {})
                   and r['balanced'][sl32]['prediction'] == r['true_label']),
    )

    # Filt_FBsel: layer from Biased_Filt (cross-val), fallback -> balanced best-layer (CV)
    bal_best_cv = str(best_params.get('Bal_Select_CV', best_params['Bal_Select'])['params']['layer'])
    results['Filt_FBsel'] = _eval_biased_filt(
        records, filt_cv_layer,
        lambda r, bl=bal_best_cv: (bl in r.get('balanced', {})
                                   and r['balanced'][bl]['prediction'] == r['true_label']),
    )

    return results


# ==================================================================
# Model loading helper
# ==================================================================

def load_models(args, device):
    """Load classification head, extractor, and codebook models."""
    head_type = getattr(args, 'head_type', 'e2v')
    if head_type == 'custom':
        e2v_head = load_custom_head(args.codebook_source, 'e2v', device)
    else:
        e2v_head = load_e2v_head(E2V_HEAD_PATH, device)
    extractor = get_emotion2vec_extractor()

    codebook_dir = Path(args.codebook_dir)
    balanced_model = load_codebook(
        str(codebook_dir / args.codebook_config / args.codebook_source / 'balanced.pt'), device
    )

    fair_map = DATASET_TO_FAIR_MAP.get(args.codebook_source, {})
    inv_fair_map = {}
    for orig, fair in fair_map.items():
        if fair in FAIR_EMOTIONS and fair not in inv_fair_map:
            inv_fair_map[fair] = orig

    biased_models = {}
    for fair_emo, orig_emo in inv_fair_map.items():
        path = codebook_dir / args.codebook_config / args.codebook_source / f'biased_{orig_emo}.pt'
        model = load_codebook(str(path), device)
        if model:
            biased_models[fair_emo] = model

    logger.info(f"Loaded codebooks: balanced={'OK' if balanced_model else 'MISSING'}, "
                f"biased={list(biased_models.keys())}")
    return e2v_head, extractor, balanced_model, biased_models


METHOD_ORDER = ['Baseline', 'Bal_L32', 'Bal_Select', 'Unfilt', 'Unfilt_ID',
                'Filt', 'Filt_ID', 'Filt_FB32', 'Filt_FBsel']


def _subdir(args, prefix):
    """Return subdirectory name aware of head-type, metric, and codebook config."""
    ht = getattr(args, 'head_type', 'e2v')
    mt = getattr(args, 'metric', 'cosine')
    cb = getattr(args, 'codebook_config', '2x32')
    suffix = f'_{ht}' if ht != 'e2v' else ''
    suffix += f'_{mt}' if mt != 'cosine' else ''
    suffix += f'_{cb}' if cb != '2x32' else ''
    return f'{prefix}{suffix}'


# ==================================================================
# Phase: val-search  (4 tasks, GPU)
# ==================================================================

def run_val_search(args, layers, device):
    """Produce two sets of tuned parameters:
      - best_params_id.json:  tuned on own VAL only  (+ Filt_ID on own TEST)
      - best_params_ood.json: tuned on own VAL+TEST + other 3 IDs' TEST
    """
    cb_source = args.codebook_source
    e2v_head, extractor, balanced_model, biased_models = load_models(args, device)

    out_dir = Path(args.output_dir) / _subdir(args, 'unified') / 'e2v' / cb_source
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.output_dir) / _subdir(args, 'sample_data') / 'e2v' / cb_source
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _serialize(best_params):
        return {m: {'val_acc': d['acc'], 'params': d['params']}
                for m, d in best_params.items()}

    # --- 1) ID params: tune on own VAL only ---
    logger.info("Collecting own VAL data...")
    t0 = time.time()
    val_data = collect_sample_data(
        cb_source, [cb_source], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='val', metric=args.metric,
    )
    val_records = val_data.get(cb_source, [])
    if not val_records:
        logger.error(f"No validation data for {cb_source}!")
        sys.exit(1)
    logger.info(f"  VAL: {len(val_records)} samples ({time.time()-t0:.0f}s)")

    with open(raw_dir / f'{cb_source}_val.json', 'w') as f:
        json.dump(val_records, f)

    logger.info("Searching best params for ID evaluation (own VAL only)...")
    t1 = time.time()
    id_params = search_best_params(val_records, layers)
    logger.info(f"  ID param search done in {time.time()-t1:.0f}s")

    # --- 1b) Unfilt_ID + Filt_ID: tune on own TEST only ---
    logger.info("Collecting own TEST data for ID-only tuning...")
    t1b = time.time()
    test_data = collect_sample_data(
        cb_source, [cb_source], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='test', metric=args.metric,
    )
    test_records = test_data.get(cb_source, [])
    logger.info(f"  TEST: {len(test_records)} samples ({time.time()-t1b:.0f}s)")

    with open(raw_dir / f'{cb_source}_test.json', 'w') as f:
        json.dump(test_records, f)

    if test_records:
        test_basic = evaluate_basic_methods(test_records, layers)

        # Unfilt_ID: best Unfilt layer on own TEST
        unfilt_id_best_layer, unfilt_id_best_acc = 1, 0
        for layer in layers:
            key = f'Unfilt-L{layer}'
            if key in test_basic and test_basic[key] > unfilt_id_best_acc:
                unfilt_id_best_acc = test_basic[key]
                unfilt_id_best_layer = layer
        id_params['Unfilt_ID'] = {'acc': unfilt_id_best_acc, 'params': {'layer': unfilt_id_best_layer}}
        logger.info(f"  Unfilt_ID: L{unfilt_id_best_layer} -> {unfilt_id_best_acc:.4%}")

        # Filt_ID: best Biased_Filt layer on own TEST
        filt_id_best_layer, filt_id_best_acc = 1, 0
        for layer in layers:
            key = f'Biased_Filt-L{layer}'
            if key in test_basic and test_basic[key] > filt_id_best_acc:
                filt_id_best_acc = test_basic[key]
                filt_id_best_layer = layer
        id_params['Filt_ID'] = {'acc': filt_id_best_acc, 'params': {'layer': filt_id_best_layer}}
        logger.info(f"  Filt_ID: L{filt_id_best_layer} -> {filt_id_best_acc:.4%}")
    else:
        id_params['Unfilt_ID'] = id_params['Unfilt']
        id_params['Filt_ID'] = id_params['Biased_Filt']
        logger.warning("  No TEST data for ID-only tuning, using CV params")

    with open(out_dir / 'best_params_id.json', 'w') as f:
        json.dump(_serialize(id_params), f, indent=2)
    logger.info(f"  Saved to best_params_id.json")

    # --- 2) OOD params: tune on own VAL+TEST + other 3 IDs' TEST ---
    logger.info("Collecting own VAL+TEST data...")
    t2 = time.time()
    vt_data = collect_sample_data(
        cb_source, [cb_source], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='val+test', metric=args.metric,
    )
    ood_tune_records = list(vt_data.get(cb_source, []))
    logger.info(f"  Own VAL+TEST: {len(ood_tune_records)} samples ({time.time()-t2:.0f}s)")

    with open(raw_dir / f'{cb_source}_val_test.json', 'w') as f:
        json.dump(ood_tune_records, f)

    other_ids = [ds for ds in ID_DATASETS if ds != cb_source]
    logger.info(f"Collecting other IDs' TEST data: {other_ids}")
    t3 = time.time()
    other_data = collect_sample_data(
        cb_source, other_ids, balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='test', metric=args.metric,
    )
    for ds, recs in other_data.items():
        ood_tune_records.extend(recs)
        logger.info(f"  + {ds} TEST: {len(recs)} samples")
        with open(raw_dir / f'{ds}_tune.json', 'w') as f:
            json.dump(recs, f)
    logger.info(f"  Total OOD tuning samples: {len(ood_tune_records)} ({time.time()-t3:.0f}s)")

    logger.info("Searching best params for OOD evaluation (all 4 IDs)...")
    t4 = time.time()
    ood_params = search_best_params(ood_tune_records, layers)
    logger.info(f"  OOD param search done in {time.time()-t4:.0f}s")

    # ID-tuned methods use same params in OOD context (always tuned on ID test only)
    ood_params['Unfilt_ID'] = id_params['Unfilt_ID']
    ood_params['Filt_ID'] = id_params['Filt_ID']

    # Bal_Select_CV: from OOD (cross-val) params, used by Filt_FBsel fallback
    ood_params['Bal_Select_CV'] = ood_params['Bal_Select']

    with open(out_dir / 'best_params_ood.json', 'w') as f:
        json.dump(_serialize(ood_params), f, indent=2)
    logger.info(f"  Saved to best_params_ood.json")


# ==================================================================
# Phase: test-apply  (4 x 7 = 28 tasks, GPU)
# ==================================================================

def run_test_apply(args, layers, device):
    """Load saved params, collect test data for ONE dataset, evaluate.

    ID evaluation  -> best_params_id.json  (tuned on ID VAL + Filt_ID on ID TEST)
    OOD evaluation -> best_params_ood.json (tuned on cross-val pool)
    """
    cb_source = args.codebook_source
    test_ds = args.test_dataset
    is_id = (test_ds == cb_source)

    if not test_ds:
        logger.error("--test-dataset required for test-apply phase")
        sys.exit(1)

    out_dir = Path(args.output_dir) / _subdir(args, 'unified') / 'e2v' / cb_source
    params_file = 'best_params_id.json' if is_id else 'best_params_ood.json'
    params_path = out_dir / params_file
    if not params_path.exists():
        logger.error(f"Params not found: {params_path}. Run val-search first.")
        sys.exit(1)

    logger.info(f"Using {params_file} ({'ID' if is_id else 'OOD'} mode)")
    with open(params_path) as f:
        serialized = json.load(f)

    best_params = {}
    for mname, mdata in serialized.items():
        best_params[mname] = {'acc': mdata['val_acc'], 'params': mdata['params']}

    e2v_head, extractor, balanced_model, biased_models = load_models(args, device)

    logger.info(f"Collecting test data for {test_ds}...")
    t0 = time.time()
    test_data = collect_sample_data(
        cb_source, [test_ds], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='test', metric=args.metric,
    )
    logger.info(f"Data collected in {time.time()-t0:.0f}s")

    records = test_data.get(test_ds, [])
    if not records:
        logger.warning(f"No test data for {test_ds}, writing empty result")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f'{test_ds}_results.json', 'w') as f:
            json.dump({'dataset': test_ds, 'n_samples': 0, 'results': {}}, f, indent=2)
        return

    raw_dir = Path(args.output_dir) / _subdir(args, 'sample_data') / 'e2v' / cb_source
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / f'{test_ds}.json', 'w') as f:
        json.dump(records, f)

    ds_results = apply_params(records, layers, best_params)

    out_dir = Path(args.output_dir) / _subdir(args, 'unified') / 'e2v' / cb_source
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'{test_ds}_results.json', 'w') as f:
        json.dump({'dataset': test_ds, 'n_samples': len(records), 'results': ds_results}, f, indent=2)

    logger.info(f"Results for {test_ds} ({len(records)} samples):")
    for m in METHOD_ORDER:
        if m in ds_results:
            logger.info(f"  {m}: {ds_results[m]*100:.2f}%")


# ==================================================================
# Phase: aggregate  (4 tasks, CPU)
# ==================================================================

def run_aggregate(args):
    """Combine per-dataset result files into a single summary."""
    cb_source = args.codebook_source
    out_dir = Path(args.output_dir) / _subdir(args, 'unified') / 'e2v' / cb_source

    id_params_path = out_dir / 'best_params_id.json'
    ood_params_path = out_dir / 'best_params_ood.json'
    for pp in [id_params_path, ood_params_path]:
        if not pp.exists():
            logger.error(f"{pp.name} not found in {out_dir}")
            sys.exit(1)

    with open(id_params_path) as f:
        id_best_params = json.load(f)
    with open(ood_params_path) as f:
        ood_best_params = json.load(f)

    source_ood = get_ood_datasets(cb_source)
    all_eval_datasets = [cb_source] + source_ood

    final_results = {
        'codebook_source': cb_source,
        'best_params_id': id_best_params,
        'best_params_ood': ood_best_params,
        'id': {},
        'en_ood': {},
    }

    missing = []
    for ds in all_eval_datasets:
        result_file = out_dir / f'{ds}_results.json'
        if not result_file.exists():
            missing.append(ds)
            continue
        with open(result_file) as f:
            ds_data = json.load(f)
        ds_results = ds_data.get('results', {})
        if not ds_results:
            continue

        if ds == cb_source:
            final_results['id'][ds] = ds_results
        elif ds in source_ood:
            final_results['en_ood'][ds] = ds_results

    if missing:
        logger.warning(f"Missing results for: {missing}")

    group_data = final_results['en_ood']
    if group_data:
        methods = set()
        for dr in group_data.values():
            methods.update(dr.keys())
        avg = {}
        for m in sorted(methods):
            vals = [group_data[ds][m] for ds in group_data if m in group_data[ds]]
            if vals:
                avg[m] = sum(vals) / len(vals)
        final_results['en_ood_avg'] = avg

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY: {cb_source}")
    print(f"{'='*80}")
    header = f"{'Method':>20s} {'ID':>8s} {'EN-OOD':>8s}"
    print(header)
    print("-" * 40)
    for m in METHOD_ORDER:
        id_acc = final_results['id'].get(cb_source, {}).get(m, 0)
        en_avg = final_results.get('en_ood_avg', {}).get(m, 0)
        print(f"{m:>20s} {id_acc*100:>7.2f}% {en_avg*100:>7.2f}%")
    print("=" * 80)


# ==================================================================
# Phase: all  (original sequential behavior)
# ==================================================================

def run_all(args, layers, device):
    """Run val-search + test-apply (all datasets) + aggregate sequentially.

    ID eval:  params tuned on own VAL (+ Filt_ID on ID TEST),  eval on ID TEST
    OOD eval: params tuned on cross-val pool,                  eval on EN OOD TEST
    """
    cb_source = args.codebook_source
    e2v_head, extractor, balanced_model, biased_models = load_models(args, device)

    # 1a) Collect own VAL -> tune ID params
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1a: Collecting own VAL data")
    logger.info("=" * 70)
    t0 = time.time()
    val_data = collect_sample_data(
        cb_source, [cb_source], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='val', metric=args.metric,
    )
    val_records = val_data.get(cb_source, [])
    if not val_records:
        logger.error(f"No validation data for {cb_source}!")
        return
    logger.info(f"  VAL: {len(val_records)} samples ({time.time()-t0:.0f}s)")

    logger.info("Tuning ID params (own VAL only)...")
    t1 = time.time()
    id_params = search_best_params(val_records, layers)
    logger.info(f"  ID param search done in {time.time()-t1:.0f}s")

    # 1a-extra) Unfilt_ID + Filt_ID: tune on own TEST only
    logger.info("Collecting own TEST data for ID-only tuning...")
    t1b = time.time()
    test_data_for_id = collect_sample_data(
        cb_source, [cb_source], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='test', metric=args.metric,
    )
    test_records_id = test_data_for_id.get(cb_source, [])
    logger.info(f"  TEST for ID tuning: {len(test_records_id)} samples ({time.time()-t1b:.0f}s)")

    if test_records_id:
        test_basic = evaluate_basic_methods(test_records_id, layers)

        unfilt_id_best_layer, unfilt_id_best_acc = 1, 0
        for layer in layers:
            key = f'Unfilt-L{layer}'
            if key in test_basic and test_basic[key] > unfilt_id_best_acc:
                unfilt_id_best_acc = test_basic[key]
                unfilt_id_best_layer = layer
        id_params['Unfilt_ID'] = {'acc': unfilt_id_best_acc, 'params': {'layer': unfilt_id_best_layer}}
        logger.info(f"  Unfilt_ID: L{unfilt_id_best_layer} -> {unfilt_id_best_acc:.4%}")

        filt_id_best_layer, filt_id_best_acc = 1, 0
        for layer in layers:
            key = f'Biased_Filt-L{layer}'
            if key in test_basic and test_basic[key] > filt_id_best_acc:
                filt_id_best_acc = test_basic[key]
                filt_id_best_layer = layer
        id_params['Filt_ID'] = {'acc': filt_id_best_acc, 'params': {'layer': filt_id_best_layer}}
        logger.info(f"  Filt_ID: L{filt_id_best_layer} -> {filt_id_best_acc:.4%}")
    else:
        id_params['Unfilt_ID'] = id_params['Unfilt']
        id_params['Filt_ID'] = id_params['Biased_Filt']

    # 1b) Collect own VAL+TEST + other 3 IDs' TEST -> tune OOD params
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1b: Collecting OOD tuning data (all 4 IDs)")
    logger.info("=" * 70)
    t2 = time.time()
    vt_data = collect_sample_data(
        cb_source, [cb_source], balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='val+test', metric=args.metric,
    )
    ood_tune_records = list(vt_data.get(cb_source, []))
    logger.info(f"  Own VAL+TEST: {len(ood_tune_records)} samples")

    other_ids = [ds for ds in ID_DATASETS if ds != cb_source]
    other_data = collect_sample_data(
        cb_source, other_ids, balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='test', metric=args.metric,
    )
    for ds, recs in other_data.items():
        ood_tune_records.extend(recs)
        logger.info(f"  + {ds} TEST: {len(recs)} samples")
    logger.info(f"  Total OOD tuning: {len(ood_tune_records)} ({time.time()-t2:.0f}s)")

    logger.info("Tuning OOD params (all 4 IDs)...")
    t3 = time.time()
    ood_params = search_best_params(ood_tune_records, layers)
    logger.info(f"  OOD param search done in {time.time()-t3:.0f}s")

    ood_params['Unfilt_ID'] = id_params['Unfilt_ID']
    ood_params['Filt_ID'] = id_params['Filt_ID']
    ood_params['Bal_Select_CV'] = ood_params['Bal_Select']

    # 2) Collect test data for all EN evaluation datasets
    source_ood = get_ood_datasets(cb_source)
    all_eval_datasets = [cb_source] + source_ood
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2: Collecting TEST data for evaluation datasets")
    logger.info("=" * 70)
    t4 = time.time()
    all_data = collect_sample_data(
        cb_source, all_eval_datasets, balanced_model, biased_models,
        e2v_head, extractor, layers, args.max_samples, device,
        split='test', metric=args.metric,
    )
    logger.info(f"  Phase 2 done in {time.time()-t4:.0f}s")

    raw_dir = Path(args.output_dir) / _subdir(args, 'sample_data') / 'e2v' / cb_source
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ds, records in all_data.items():
        with open(raw_dir / f'{ds}.json', 'w') as f:
            json.dump(records, f)

    # 3) Apply params + aggregate
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3: Applying params (ID params->ID, OOD params->OOD)")
    logger.info("=" * 70)

    final_results = {
        'codebook_source': cb_source,
        'best_params_id': {m: {'val_acc': d['acc'], 'params': d['params']}
                           for m, d in id_params.items()},
        'best_params_ood': {m: {'val_acc': d['acc'], 'params': d['params']}
                            for m, d in ood_params.items()},
        'id': {},
        'en_ood': {},
    }

    for ds, records in all_data.items():
        if not records:
            continue
        is_id = (ds == cb_source)
        params = id_params if is_id else ood_params
        ds_results = apply_params(records, layers, params)
        if is_id:
            final_results['id'][ds] = ds_results
        elif ds in source_ood:
            final_results['en_ood'][ds] = ds_results

    group_data = final_results['en_ood']
    if group_data:
        methods = set()
        for dr in group_data.values():
            methods.update(dr.keys())
        avg = {}
        for m in sorted(methods):
            vals = [group_data[ds][m] for ds in group_data if m in group_data[ds]]
            if vals:
                avg[m] = sum(vals) / len(vals)
        final_results['en_ood_avg'] = avg

    out_dir = Path(args.output_dir) / _subdir(args, 'unified') / 'e2v'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{cb_source}.json'
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY: {cb_source}")
    print(f"{'='*80}")
    header = f"{'Method':>20s} {'ID':>8s} {'EN-OOD':>8s}"
    print(header)
    print("-" * 40)
    for m in METHOD_ORDER:
        id_acc = final_results['id'].get(cb_source, {}).get(m, 0)
        en_avg = final_results.get('en_ood_avg', {}).get(m, 0)
        print(f"{m:>20s} {id_acc*100:>7.2f}% {en_avg*100:>7.2f}%")
    print("=" * 80)


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation')
    parser.add_argument('--codebook-source', type=str, required=True,
                        choices=ID_DATASETS)
    parser.add_argument('--codebook-dir', type=str, default=str(CODEBOOK_DIR / 'e2v'))
    parser.add_argument('--codebook-config', type=str, default='2x32')
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--phase', type=str, default='all',
                        choices=['val-search', 'test-apply', 'aggregate', 'all'])
    parser.add_argument('--test-dataset', type=str,
                        help='Single dataset name (required for test-apply phase)')
    parser.add_argument('--head-type', type=str, default='e2v',
                        choices=['e2v', 'custom'],
                        help='Classification head: e2v (official) or custom (per-source trained)')
    parser.add_argument('--metric', type=str, default='l2',
                        choices=['cosine', 'l2'],
                        help='Similarity metric: cosine or l2 (negative L2 distance)')
    args = parser.parse_args()

    set_seed()
    cb_cfg = args.codebook_config  # e.g. '2x32', '128x8'
    num_layers = int(cb_cfg.split('x')[1]) if 'x' in cb_cfg else 32
    layers = list(range(1, num_layers + 1))
    device = args.device

    if args.phase != 'aggregate':
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = 'cpu'

    logger.info("=" * 70)
    logger.info(f"Unified Evaluation -- phase={args.phase}, source={args.codebook_source}, metric={args.metric}")
    if args.test_dataset:
        logger.info(f"  test-dataset: {args.test_dataset}")
    logger.info("=" * 70)

    if args.phase == 'val-search':
        run_val_search(args, layers, device)
    elif args.phase == 'test-apply':
        run_test_apply(args, layers, device)
    elif args.phase == 'aggregate':
        run_aggregate(args)
    else:
        run_all(args, layers, device)


if __name__ == '__main__':
    main()
