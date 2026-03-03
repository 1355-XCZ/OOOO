#!/usr/bin/env python3
"""
RQ4 Comprehensive Evaluator with Ratio Codebooks

Extends RQ4 evaluation to include ratio (mixed) codebooks alongside
balanced and biased codebooks.  Saves per-sample data for every codebook
type so that tables/figures can be generated flexibly.

Codebook types evaluated:
  balanced, biased, mixed_r50, mixed_r70, mixed_r80, mixed_r95, mixed_r99

For each (config, source) pair the script:
  1. Extracts features once for all test samples.
  2. For every codebook type, computes per-layer cosine similarity
     and E2V head predictions.
  3. Saves comprehensive per-sample JSON.
  4. Runs the 9 RQ4 evaluation methods for each codebook type that can
     serve as the "emotion-specific" component, producing a method×type
     result matrix.

Output layout:
  results/rq4_ratio/{config}/{source}/sample_data.json
  results/rq4_ratio/{config}/{source}/methods.json
  results/rq4_ratio/{config}/{source}/params.json

Usage:
  python -m paper_pipeline.evaluators.rq4_ratio_evaluate \
      --codebook-source esd_en --codebook-config 2x32
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.dataset_config import DATASET_CONFIGS
from core.config import (
    E2V_LABELS, E2V_HEAD_PATH,
    FAIR_EMOTIONS, FAIR_E2V_INDICES, DATASET_TO_FAIR_MAP,
    CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR,
    set_seed,
)
from core.quantize import load_codebook, compute_similarity, get_all_reconstructions
from core.classify import E2VClassificationHead, load_e2v_head
from core.features import get_emotion2vec_extractor, extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']
EN_OOD_DATASETS = [
    'msp',
    'cameo_emns', 'cameo_enterface', 'cameo_jl_corpus',
]

BIASED_EMOTIONS = FAIR_EMOTIONS

CODEBOOK_TYPES = ['biased', 'mixed_r99']

METHOD_ORDER = [
    'Baseline', 'Bal_LS', 'Bal_Select',
    'Unfilt', 'Unfilt_ID',
    'Filt', 'Filt_ID', 'Filt_FB_LS', 'Filt_FBsel',
]


def _fname_for_type(cb_type: str, emotion: str) -> str:
    if cb_type == 'biased':
        return f'biased_{emotion}.pt'
    return f'{cb_type.replace("mixed_r", "mixed_" + emotion + "_r")}.pt'


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_all_codebooks(
    cb_config: str,
    cb_source: str,
    device: str,
) -> dict:
    """Load balanced + all emotion-specific codebook types.

    Returns:
        {
            'balanced': model,
            'biased':   {'angry': model, ...},
            'mixed_r50': {'angry': model, ...},
            ...
        }
    """
    cb_dir = CODEBOOK_DIR / 'e2v' / cb_config / cb_source

    result = {}

    balanced = load_codebook(str(cb_dir / 'balanced.pt'), device)
    result['balanced'] = balanced

    fair_map = DATASET_TO_FAIR_MAP.get(cb_source, {})
    inv_map = {}
    for orig, fair in fair_map.items():
        if fair in FAIR_EMOTIONS and fair not in inv_map:
            inv_map[fair] = orig

    for cb_type in CODEBOOK_TYPES:
        type_models = {}
        for fair_emo, orig_emo in inv_map.items():
            fname = _fname_for_type(cb_type, orig_emo)
            path = cb_dir / fname
            model = load_codebook(str(path), device)
            if model:
                type_models[fair_emo] = model
            else:
                logger.warning(f"Missing codebook: {path}")
        result[cb_type] = type_models
        logger.info(f"  {cb_type}: loaded {list(type_models.keys())}")

    return result


# ------------------------------------------------------------------
# Sample data collection (GPU)
# ------------------------------------------------------------------

def collect_comprehensive_data(
    cb_source: str,
    test_datasets: List[str],
    all_codebooks: dict,
    e2v_head: E2VClassificationHead,
    extractor,
    layers: List[int],
    max_samples: int,
    device: str,
    split: str = 'test',
) -> Dict[str, List[dict]]:
    """Collect per-sample predictions from ALL codebook types at ALL layers."""

    all_data = {}
    balanced_model = all_codebooks['balanced']

    for test_ds in test_datasets:
        fair_map = DATASET_TO_FAIR_MAP.get(test_ds)
        if not fair_map:
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

            # Baseline
            with torch.no_grad():
                logits = e2v_head(features_gpu.unsqueeze(0) if features_gpu.dim() == 2 else features_gpu)
                valid_logits = logits[:, FAIR_E2V_INDICES]
                probs = F.softmax(valid_logits, dim=-1).squeeze(0).cpu().numpy()

            bl_pred = FAIR_EMOTIONS[int(np.argmax(probs))]
            record['baseline'] = {
                'softmax': {emo: round(float(probs[i]), 6) for i, emo in enumerate(FAIR_EMOTIONS)},
                'prediction': bl_pred,
                'correct': bl_pred == true_label,
            }

            # Balanced
            if balanced_model is not None:
                bal_recons = get_all_reconstructions(balanced_model, features_gpu, layers, device)
                bal_data = {}
                for layer in layers:
                    if layer in bal_recons:
                        sim = compute_similarity(features_gpu, bal_recons[layer], 'cosine')
                        with torch.no_grad():
                            r_logits = e2v_head(
                                bal_recons[layer].unsqueeze(0) if bal_recons[layer].dim() == 2 else bal_recons[layer])
                            r_probs = F.softmax(r_logits[:, FAIR_E2V_INDICES], dim=-1).squeeze(0).cpu().numpy()
                        bal_pred = FAIR_EMOTIONS[int(np.argmax(r_probs))]
                        bal_data[str(layer)] = {
                            'cosine': round(sim, 6),
                            'prediction': bal_pred,
                        }
                record['balanced'] = bal_data

            # Emotion-specific codebook types
            for cb_type in CODEBOOK_TYPES:
                type_models = all_codebooks.get(cb_type, {})
                if not type_models:
                    continue
                type_data = {}
                for emo, model in type_models.items():
                    emo_recons = get_all_reconstructions(model, features_gpu, layers, device)
                    emo_data = {}
                    for layer in layers:
                        if layer in emo_recons:
                            sim = compute_similarity(features_gpu, emo_recons[layer], 'cosine')
                            with torch.no_grad():
                                r_logits = e2v_head(
                                    emo_recons[layer].unsqueeze(0) if emo_recons[layer].dim() == 2 else emo_recons[layer])
                                r_probs = F.softmax(r_logits[:, FAIR_E2V_INDICES], dim=-1).squeeze(0).cpu().numpy()
                            cls_pred = FAIR_EMOTIONS[int(np.argmax(r_probs))]
                            emo_data[str(layer)] = {
                                'cosine': round(sim, 6),
                                'cls_prediction': cls_pred,
                            }
                    type_data[emo] = emo_data
                record[cb_type] = type_data

            ds_records.append(record)

        all_data[test_ds] = ds_records
        logger.info(f"  {test_ds}: collected {len(ds_records)} records")

    return all_data


# ------------------------------------------------------------------
# Method evaluation (CPU, from cached sample data)
# ------------------------------------------------------------------

def _eval_balanced_layer(records, layer_str):
    N = len(records)
    if N == 0:
        return 0.0
    return sum(
        1 for r in records
        if layer_str in r.get('balanced', {})
        and r['balanced'][layer_str]['prediction'] == r['true_label']
    ) / N


def _eval_unfilt(records, layer_str, cb_type_key):
    """Argmax cosine among emotion-specific codebooks → predicted emotion."""
    N = len(records)
    if N == 0:
        return 0.0
    correct = 0
    for r in records:
        emo_data = r.get(cb_type_key, {})
        best_emo, best_cos = None, -float('inf')
        for emo, ed in emo_data.items():
            if layer_str in ed and ed[layer_str]['cosine'] > best_cos:
                best_cos = ed[layer_str]['cosine']
                best_emo = emo
        if best_emo == r['true_label']:
            correct += 1
    return correct / N


def _eval_filt(records, layer_str, cb_type_key, fallback_fn):
    """Delta-cosine (emotion_cos - balanced_cos) with fallback."""
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
        emo_data = r.get(cb_type_key, {})
        best_emo, best_d = None, -float('inf')
        for emo, ed in emo_data.items():
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


def search_best_params(records, layers, cb_type_key):
    """Find best layer for each method using a given codebook type."""
    N = len(records)
    if N == 0:
        return {}

    best = {}

    bl_correct = sum(1 for r in records if r['baseline']['correct'])
    best['Baseline'] = {'acc': bl_correct / N, 'params': {}}

    # Bal_LS (last layer)
    max_l = max(layers)
    best['Bal_LS'] = {'acc': _eval_balanced_layer(records, str(max_l)), 'params': {'layer': max_l}}

    # Bal_Select
    bal_best_l, bal_best_acc = 1, 0
    for l in layers:
        acc = _eval_balanced_layer(records, str(l))
        if acc > bal_best_acc:
            bal_best_acc = acc
            bal_best_l = l
    best['Bal_Select'] = {'acc': bal_best_acc, 'params': {'layer': bal_best_l}}

    # Unfilt
    unfilt_best_l, unfilt_best_acc = 1, 0
    for l in layers:
        acc = _eval_unfilt(records, str(l), cb_type_key)
        if acc > unfilt_best_acc:
            unfilt_best_acc = acc
            unfilt_best_l = l
    best['Unfilt'] = {'acc': unfilt_best_acc, 'params': {'layer': unfilt_best_l}}

    # Filt (fallback → baseline)
    filt_best_l, filt_best_acc = 1, 0
    for l in layers:
        acc = _eval_filt(records, str(l), cb_type_key, lambda r: r['baseline']['correct'])
        if acc > filt_best_acc:
            filt_best_acc = acc
            filt_best_l = l
    best['Biased_Filt'] = {'acc': filt_best_acc, 'params': {'layer': filt_best_l}}

    return best


def apply_params(records, layers, best_params, cb_type_key):
    """Apply tuned parameters for one codebook type and produce method accuracies."""
    N = len(records)
    if N == 0:
        return {}

    results = {}
    results['Baseline'] = sum(1 for r in records if r['baseline']['correct']) / N

    # Bal_LS
    sl_ls = str(best_params['Bal_LS']['params']['layer'])
    results['Bal_LS'] = _eval_balanced_layer(records, sl_ls)

    # Bal_Select
    sl_sel = str(best_params['Bal_Select']['params']['layer'])
    results['Bal_Select'] = _eval_balanced_layer(records, sl_sel)

    # Unfilt (CV layer)
    unfilt_l = str(best_params['Unfilt']['params']['layer'])
    results['Unfilt'] = _eval_unfilt(records, unfilt_l, cb_type_key)

    # Unfilt_ID
    unfilt_id_l = str(best_params['Unfilt_ID']['params']['layer'])
    results['Unfilt_ID'] = _eval_unfilt(records, unfilt_id_l, cb_type_key)

    # Filt (CV layer, fallback → baseline)
    filt_cv_l = str(best_params['Biased_Filt']['params']['layer'])
    results['Filt'] = _eval_filt(
        records, filt_cv_l, cb_type_key, lambda r: r['baseline']['correct'])

    # Filt_ID (ID layer, fallback → baseline)
    filt_id_l = str(best_params['Filt_ID']['params']['layer'])
    results['Filt_ID'] = _eval_filt(
        records, filt_id_l, cb_type_key, lambda r: r['baseline']['correct'])

    # Filt_FB_LS (CV layer, fallback → balanced last-layer)
    results['Filt_FB_LS'] = _eval_filt(
        records, filt_cv_l, cb_type_key,
        lambda r: (sl_ls in r.get('balanced', {})
                   and r['balanced'][sl_ls]['prediction'] == r['true_label']))

    # Filt_FBsel (CV layer, fallback → balanced best-layer)
    bal_best_cv = str(best_params.get('Bal_Select_CV', best_params['Bal_Select'])['params']['layer'])
    results['Filt_FBsel'] = _eval_filt(
        records, filt_cv_l, cb_type_key,
        lambda r, bl=bal_best_cv: (bl in r.get('balanced', {})
                                   and r['balanced'][bl]['prediction'] == r['true_label']))

    return results


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(args):
    cb_source = args.codebook_source
    cb_config = args.codebook_config
    num_layers = int(cb_config.split('x')[1]) if 'x' in cb_config else 32
    layers = list(range(1, num_layers + 1))
    device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    out_dir = RESULTS_DIR / 'rq4_ratio' / cb_config / cb_source
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and (out_dir / 'methods.json').exists():
        logger.info(f"Results exist at {out_dir / 'methods.json'}, use --force to overwrite")
        return

    logger.info(f"RQ4 Ratio Evaluate: config={cb_config}, source={cb_source}")

    e2v_head = load_e2v_head(E2V_HEAD_PATH, device)
    extractor = get_emotion2vec_extractor()

    logger.info("Loading all codebooks...")
    all_codebooks = load_all_codebooks(cb_config, cb_source, device)

    # 1) Collect VAL data for parameter tuning
    logger.info("Phase 1: Collecting VAL data for parameter tuning...")
    t0 = time.time()
    val_data = collect_comprehensive_data(
        cb_source, [cb_source], all_codebooks, e2v_head, extractor,
        layers, args.max_samples, device, split='val',
    )
    val_records = val_data.get(cb_source, [])
    logger.info(f"  VAL: {len(val_records)} samples ({time.time()-t0:.0f}s)")

    # Also collect own TEST for ID-tuned methods
    logger.info("Collecting own TEST for ID-tuned methods...")
    t1 = time.time()
    id_test_data = collect_comprehensive_data(
        cb_source, [cb_source], all_codebooks, e2v_head, extractor,
        layers, args.max_samples, device, split='test',
    )
    id_test_records = id_test_data.get(cb_source, [])
    logger.info(f"  TEST: {len(id_test_records)} samples ({time.time()-t1:.0f}s)")

    # 2) Collect OOD tuning pool (own VAL+TEST + other IDs' TEST)
    logger.info("Phase 2: Collecting OOD tuning data...")
    t2 = time.time()
    vt_data = collect_comprehensive_data(
        cb_source, [cb_source], all_codebooks, e2v_head, extractor,
        layers, args.max_samples, device, split='val+test',
    )
    ood_tune = list(vt_data.get(cb_source, []))

    other_ids = [ds for ds in ID_DATASETS if ds != cb_source]
    other_data = collect_comprehensive_data(
        cb_source, other_ids, all_codebooks, e2v_head, extractor,
        layers, args.max_samples, device, split='test',
    )
    for ds, recs in other_data.items():
        ood_tune.extend(recs)
        logger.info(f"  + {ds}: {len(recs)} samples")
    logger.info(f"  OOD tuning pool: {len(ood_tune)} ({time.time()-t2:.0f}s)")

    # 3) Collect test data for all EN OOD datasets
    all_eval_datasets = [cb_source] + EN_OOD_DATASETS
    logger.info("Phase 3: Collecting TEST data for all eval datasets...")
    t3 = time.time()
    all_test_data = collect_comprehensive_data(
        cb_source, all_eval_datasets, all_codebooks, e2v_head, extractor,
        layers, args.max_samples, device, split='test',
    )
    logger.info(f"  Phase 3 done ({time.time()-t3:.0f}s)")

    # Save sample data
    sample_out = {}
    for ds, recs in all_test_data.items():
        sample_out[ds] = recs
    with open(out_dir / 'sample_data.json', 'w') as f:
        json.dump(sample_out, f)
    logger.info(f"  Saved sample data: {out_dir / 'sample_data.json'}")

    # 4) Tune + apply for each codebook type
    all_params = {}
    method_results = {}

    for cb_type in CODEBOOK_TYPES:
        logger.info(f"\n--- Evaluating methods for: {cb_type} ---")

        # ID params from VAL
        id_params = search_best_params(val_records, layers, cb_type)

        # ID params: Unfilt_ID + Filt_ID from own TEST
        if id_test_records:
            unfilt_id_best_l, unfilt_id_best_acc = 1, 0
            for l in layers:
                acc = _eval_unfilt(id_test_records, str(l), cb_type)
                if acc > unfilt_id_best_acc:
                    unfilt_id_best_acc = acc
                    unfilt_id_best_l = l
            id_params['Unfilt_ID'] = {'acc': unfilt_id_best_acc, 'params': {'layer': unfilt_id_best_l}}

            filt_id_best_l, filt_id_best_acc = 1, 0
            for l in layers:
                acc = _eval_filt(id_test_records, str(l), cb_type, lambda r: r['baseline']['correct'])
                if acc > filt_id_best_acc:
                    filt_id_best_acc = acc
                    filt_id_best_l = l
            id_params['Filt_ID'] = {'acc': filt_id_best_acc, 'params': {'layer': filt_id_best_l}}
        else:
            id_params['Unfilt_ID'] = id_params['Unfilt']
            id_params['Filt_ID'] = id_params['Biased_Filt']

        # OOD params from tuning pool
        ood_params = search_best_params(ood_tune, layers, cb_type)
        ood_params['Unfilt_ID'] = id_params['Unfilt_ID']
        ood_params['Filt_ID'] = id_params['Filt_ID']
        ood_params['Bal_Select_CV'] = ood_params['Bal_Select']

        all_params[cb_type] = {
            'id': {m: {'acc': d['acc'], 'params': d['params']} for m, d in id_params.items()},
            'ood': {m: {'acc': d['acc'], 'params': d['params']} for m, d in ood_params.items()},
        }

        # Apply to all datasets
        type_results = {'id': {}, 'ood': {}}
        for ds, recs in all_test_data.items():
            if not recs:
                continue
            is_id = (ds == cb_source)
            params = id_params if is_id else ood_params
            ds_acc = apply_params(recs, layers, params, cb_type)
            if is_id:
                type_results['id'][ds] = ds_acc
            else:
                type_results['ood'][ds] = ds_acc

        # OOD average
        ood_group = type_results['ood']
        if ood_group:
            methods = set()
            for dr in ood_group.values():
                methods.update(dr.keys())
            avg = {}
            for m in sorted(methods):
                vals = [ood_group[ds][m] for ds in ood_group if m in ood_group[ds]]
                if vals:
                    avg[m] = sum(vals) / len(vals)
            type_results['ood_avg'] = avg

        method_results[cb_type] = type_results

        # Summary
        id_acc = type_results['id'].get(cb_source, {})
        ood_avg = type_results.get('ood_avg', {})
        logger.info(f"  {cb_type} summary:")
        for m in METHOD_ORDER:
            id_v = id_acc.get(m, 0) * 100
            ood_v = ood_avg.get(m, 0) * 100
            logger.info(f"    {m:>15s}  ID={id_v:6.2f}%  OOD={ood_v:6.2f}%")

    # Save params and method results
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    with open(out_dir / 'methods.json', 'w') as f:
        json.dump(method_results, f, indent=2)

    logger.info(f"\nAll results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='RQ4 Ratio Comprehensive Evaluator')
    parser.add_argument('--codebook-source', type=str, required=True, choices=ID_DATASETS)
    parser.add_argument('--codebook-config', type=str, required=True,
                        help='Codebook config, e.g., 2x32, 128x8')
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--force', action='store_true', help='Overwrite existing results')
    args = parser.parse_args()

    set_seed()
    run(args)


if __name__ == '__main__':
    main()
