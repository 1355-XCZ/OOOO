#!/usr/bin/env python3
"""
RQ2.1 Evaluator -- SER metrics for Matched/Unmatched/All/Balanced codebooks

Computes per-layer SER metrics (F1-macro, recall-macro, accuracy) using the
appropriate classification head on features quantized by four codebook types:

  - balanced:          single balanced codebook for all samples
  - biased_matched:    for each emotion E, use biased_E codebook on E samples
  - biased_unmatched:  for each emotion E, use biased_{C!=E} codebooks on E samples,
                       pool predictions across all non-matching codebooks
  - biased_all:        each biased codebook tested on ALL emotions fairly,
                       per-codebook metrics averaged across 4 codebooks

Also outputs per-codebook × per-emotion × per-layer recall for the RQ2 grid figure.

Supports multiple SSL models (e2v, hubert, wavlm) and codebook configs.

Output per (ssl_model, codebook_dataset, test_dataset) pair:
  results/rq2_matched_ser_{config}/{ssl_model}/{src}_to_{tgt}_ood.json

Can be run standalone:
  python -m paper_pipeline.evaluators.rq2_1_matched_ser \
      --ssl-model e2v --codebook-config 2x24 \
      --codebook-dataset esd_en --test-dataset iemocap
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import DEFAULT_MAX_SAMPLES
from configs.dataset_config import DATASET_CONFIGS
from core.config import (
    E2V_LABELS, E2V_HEAD_PATH, CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR,
)
from core.features import get_ssl_extractor, extract_features
from core.quantize import load_codebook, compute_similarity
from core.classify import load_e2v_head, load_custom_head

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']

DEFAULT_SSL_MODEL = 'e2v'
DEFAULT_CB_CONFIG = '2x24'
DEFAULT_NUM_LAYERS = 24


def _result_subdir(config: str, ssl_model: str) -> str:
    return f'rq2_matched_ser_{config}/{ssl_model}'


# ------------------------------------------------------------------
# Feature pre-extraction
# ------------------------------------------------------------------

def pre_extract_features(
    extractor,
    samples: List[Tuple[str, str]],
) -> List[Tuple[torch.Tensor, str]]:
    """Extract features for (audio_path, e2v_label) pairs. Returns cached list."""
    cached = []
    for audio_path, label in tqdm(samples, desc="Pre-extracting features"):
        feats = extract_features(extractor, audio_path)
        if feats is not None:
            cached.append((feats, label))
    logger.info(f"Cached {len(cached)}/{len(samples)} samples")
    return cached


# ------------------------------------------------------------------
# Per-layer quantize -> classify
# ------------------------------------------------------------------

def _quantize_and_classify_layer(
    model, features: torch.Tensor, layer: int,
    partial: torch.Tensor, indices: torch.Tensor,
    e2v_head,
    valid_indices: List[int],
    valid_e2v_labels: List[str],
) -> Tuple[str, torch.Tensor]:
    """Accumulate layer reconstruction and return predicted label + updated partial."""
    layer_idx = indices[:, :, layer - 1]
    cb_weights = model.rvq.layers[layer - 1]._codebook.embed[0]
    partial = partial + cb_weights[layer_idx]
    quantized = partial.squeeze(0)

    q_mean = quantized.mean(dim=0).unsqueeze(0)
    logits = e2v_head(q_mean)
    valid_logits = logits[:, valid_indices]
    probs = F.softmax(valid_logits, dim=-1).squeeze(0)
    pred_label = valid_e2v_labels[probs.argmax().item()]
    return pred_label, partial


def evaluate_codebook_on_samples(
    model,
    cached_samples: List[Tuple[torch.Tensor, str]],
    e2v_head,
    valid_indices: List[int],
    valid_e2v_labels: List[str],
    num_layers: int,
    device: str,
    desc: str = "",
) -> Dict[int, List[Tuple[str, str]]]:
    """Evaluate a single codebook on a list of (features, true_label) samples."""
    layer_pairs: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    for features, true_label in tqdm(cached_samples, desc=desc, leave=False):
        orig = features.to(device)
        feat_input = orig.unsqueeze(0) if orig.dim() == 2 else orig

        with torch.no_grad():
            _, indices, _, _ = model(feat_input)
            partial = torch.zeros_like(feat_input)

            for layer in range(1, num_layers + 1):
                pred_label, partial = _quantize_and_classify_layer(
                    model, orig, layer, partial, indices,
                    e2v_head, valid_indices, valid_e2v_labels,
                )
                layer_pairs[layer].append((true_label, pred_label))

    return dict(layer_pairs)


def compute_metrics(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """Compute accuracy, F1-macro, and recall-macro from (true, pred) label pairs."""
    if not pairs:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'recall_macro': 0.0, 'num_samples': 0}
    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'num_samples': len(pairs),
    }


def compute_per_emotion_recall(
    pairs: List[Tuple[str, str]], emotions: List[str],
) -> Dict[str, float]:
    """Compute per-class recall for each emotion from (true, pred) pairs."""
    result = {}
    for emo in emotions:
        true_count = sum(1 for t, _ in pairs if t == emo)
        if true_count == 0:
            result[emo] = 0.0
        else:
            correct = sum(1 for t, p in pairs if t == emo and p == emo)
            result[emo] = correct / true_count
    return result


# ------------------------------------------------------------------
# Matched / Unmatched / Balanced evaluation
# ------------------------------------------------------------------

def evaluate_matched_unmatched_balanced(
    balanced_codebook,
    biased_codebooks: Dict[str, object],
    cached_per_emotion: Dict[str, List[Tuple[torch.Tensor, str]]],
    e2v_head,
    valid_indices: List[int],
    valid_e2v_labels: List[str],
    num_layers: int = 32,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    eval_emotions = sorted(set(cached_per_emotion.keys()) & set(biased_codebooks.keys()))
    filtered_samples = []
    for emo in eval_emotions:
        filtered_samples.extend(cached_per_emotion[emo])

    logger.info(f"Evaluation restricted to {len(eval_emotions)} emotions with biased codebooks: {eval_emotions}")
    logger.info(f"Total samples: {len(filtered_samples)}")

    results = {}

    # --- Balanced ---
    balanced_lp = None
    if balanced_codebook is not None:
        logger.info("Evaluating balanced codebook...")
        balanced_lp = evaluate_codebook_on_samples(
            balanced_codebook, filtered_samples, e2v_head,
            valid_indices, valid_e2v_labels, num_layers, device,
            desc="Balanced",
        )
        results['balanced'] = {
            f'layer_{l}': compute_metrics(balanced_lp.get(l, []))
            for l in range(1, num_layers + 1)
        }

    # --- Biased Matched ---
    logger.info("Evaluating matched biased codebooks...")
    matched_layer_pairs: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for emotion in eval_emotions:
        samples = cached_per_emotion[emotion]
        layer_pairs = evaluate_codebook_on_samples(
            biased_codebooks[emotion], samples, e2v_head,
            valid_indices, valid_e2v_labels, num_layers, device,
            desc=f"Matched-{emotion}",
        )
        for l, pairs in layer_pairs.items():
            matched_layer_pairs[l].extend(pairs)

    results['biased_matched'] = {
        f'layer_{l}': compute_metrics(matched_layer_pairs.get(l, []))
        for l in range(1, num_layers + 1)
    }

    # --- Biased Unmatched ---
    logger.info("Evaluating unmatched biased codebooks...")
    unmatched_layer_pairs: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for emotion in eval_emotions:
        samples = cached_per_emotion[emotion]
        non_matching = [e for e in eval_emotions if e != emotion]
        for other_emo in non_matching:
            layer_pairs = evaluate_codebook_on_samples(
                biased_codebooks[other_emo], samples, e2v_head,
                valid_indices, valid_e2v_labels, num_layers, device,
                desc=f"Unmatched-{other_emo}-on-{emotion}",
            )
            for l, pairs in layer_pairs.items():
                unmatched_layer_pairs[l].extend(pairs)

    results['biased_unmatched'] = {
        f'layer_{l}': compute_metrics(unmatched_layer_pairs.get(l, []))
        for l in range(1, num_layers + 1)
    }

    # --- Biased All (fair test: each biased codebook on ALL emotions, avg across codebooks) ---
    logger.info("Evaluating biased_all (each biased codebook on all emotions)...")
    biased_all_per_cb: Dict[str, Dict[int, List[Tuple[str, str]]]] = {}
    for emotion in eval_emotions:
        layer_pairs = evaluate_codebook_on_samples(
            biased_codebooks[emotion], filtered_samples, e2v_head,
            valid_indices, valid_e2v_labels, num_layers, device,
            desc=f"BiasedAll-{emotion}",
        )
        biased_all_per_cb[emotion] = layer_pairs

    biased_all_results = {}
    for layer in range(1, num_layers + 1):
        per_cb_metrics = []
        for emotion in eval_emotions:
            pairs = biased_all_per_cb.get(emotion, {}).get(layer, [])
            per_cb_metrics.append(compute_metrics(pairs))
        biased_all_results[f'layer_{layer}'] = {
            'accuracy': float(np.mean([m['accuracy'] for m in per_cb_metrics])),
            'f1_macro': float(np.mean([m['f1_macro'] for m in per_cb_metrics])),
            'recall_macro': float(np.mean([m['recall_macro'] for m in per_cb_metrics])),
            'num_samples': sum(m['num_samples'] for m in per_cb_metrics),
        }
    results['biased_all'] = biased_all_results

    # --- Per-codebook × per-emotion × per-layer recall (for RQ2 grid figure) ---
    pcr = {}
    if balanced_lp is not None:
        pcr['balanced'] = {
            f'layer_{l}': compute_per_emotion_recall(balanced_lp.get(l, []), eval_emotions)
            for l in range(1, num_layers + 1)
        }
    for cb_emo in eval_emotions:
        cb_lp = biased_all_per_cb.get(cb_emo, {})
        pcr[f'biased_{cb_emo}'] = {
            f'layer_{l}': compute_per_emotion_recall(cb_lp.get(l, []), eval_emotions)
            for l in range(1, num_layers + 1)
        }
    results['per_codebook_per_emotion_recall'] = pcr

    return results


# ------------------------------------------------------------------
# Single (source, target) evaluation
# ------------------------------------------------------------------

def evaluate_pair(
    codebook_dataset: str,
    test_dataset: str,
    ssl_model: str = DEFAULT_SSL_MODEL,
    cb_config: str = DEFAULT_CB_CONFIG,
    max_samples: int = 200,
    num_layers: int = DEFAULT_NUM_LAYERS,
    device: str = 'cuda',
) -> Dict:
    """Evaluate one (codebook_dataset, test_dataset) pair. Returns result dict."""
    codebook_base_dir = CODEBOOK_DIR / ssl_model / cb_config

    is_id = (codebook_dataset == test_dataset)
    eval_type = "ID" if is_id else "OOD"

    cb_config_obj = DATASET_CONFIGS[codebook_dataset]
    test_config = DATASET_CONFIGS[test_dataset]

    cb_e2v_map = cb_config_obj.emotion_to_e2v
    test_e2v_map = test_config.emotion_to_e2v

    cb_e2v_labels = {cb_e2v_map[e] for e in cb_config_obj.emotions if e in cb_e2v_map}
    test_e2v_labels = {test_e2v_map[e] for e in test_config.emotions if e in test_e2v_map}
    common_e2v = sorted(cb_e2v_labels & test_e2v_labels)

    if not common_e2v:
        logger.error(f"No overlapping emotions between {codebook_dataset} and {test_dataset}")
        return {}

    e2v_to_cb_emotion = {cb_e2v_map[e]: e for e in cb_config_obj.emotions if cb_e2v_map.get(e) in common_e2v}
    e2v_to_test_emotion = {test_e2v_map[e]: e for e in test_config.emotions if test_e2v_map.get(e) in common_e2v}

    logger.info("=" * 60)
    logger.info(f"RQ2 SER-F1 [{ssl_model}] ({eval_type}): {codebook_dataset} -> {test_dataset}")
    logger.info(f"  Config: {cb_config}, Layers: {num_layers}")
    logger.info(f"  Common E2V emotions: {common_e2v}")
    logger.info("=" * 60)

    with open(SPLITS_DIR / test_dataset / 'test.json') as f:
        test_splits = json.load(f)

    cb_dir = codebook_base_dir / codebook_dataset
    balanced_codebook = load_codebook(str(cb_dir / 'balanced.pt'), device)

    biased_codebooks = {}
    for e2v_label in common_e2v:
        cb_emotion = e2v_to_cb_emotion[e2v_label]
        path = cb_dir / f'biased_{cb_emotion}.pt'
        cb = load_codebook(str(path), device)
        if cb:
            biased_codebooks[e2v_label] = cb

    # Load classification head
    if ssl_model == 'e2v':
        head = load_e2v_head(E2V_HEAD_PATH, device)
    else:
        head = load_custom_head(source=codebook_dataset, ssl_model=ssl_model, device=device)

    valid_indices = [E2V_LABELS.index(l) for l in common_e2v if l in E2V_LABELS]
    valid_e2v_labels = [l for l in common_e2v if l in E2V_LABELS]

    extractor = get_ssl_extractor(ssl_model, device=device)
    cached_per_emotion: Dict[str, List[Tuple[torch.Tensor, str]]] = {}

    for e2v_label in common_e2v:
        test_emotion = e2v_to_test_emotion[e2v_label]
        files = test_splits.get(test_emotion, [])
        if not files:
            continue
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]
        samples = [(f, e2v_label) for f in files]
        cached = pre_extract_features(extractor, samples)
        if cached:
            cached_per_emotion[e2v_label] = cached

    del extractor
    torch.cuda.empty_cache()

    total = sum(len(v) for v in cached_per_emotion.values())
    logger.info(f"Feature extraction complete: {total} samples across {len(cached_per_emotion)} emotions")

    eval_results = evaluate_matched_unmatched_balanced(
        balanced_codebook=balanced_codebook,
        biased_codebooks=biased_codebooks,
        cached_per_emotion=cached_per_emotion,
        e2v_head=head,
        valid_indices=valid_indices,
        valid_e2v_labels=valid_e2v_labels,
        num_layers=num_layers,
        device=device,
    )

    return {
        'config': {
            'ssl_model': ssl_model,
            'codebook_config': cb_config,
            'codebook_dataset': codebook_dataset,
            'test_dataset': test_dataset,
            'eval_type': eval_type,
            'num_layers': num_layers,
            'max_samples': max_samples,
            'common_e2v_emotions': common_e2v,
        },
        **eval_results,
    }


# ------------------------------------------------------------------
# run() -- called by pipeline.py
# ------------------------------------------------------------------

def run(dry_run: bool = False):
    """Evaluate all ID + OOD pairs for default config (e2v, 2x24)."""
    subdir = _result_subdir(DEFAULT_CB_CONFIG, DEFAULT_SSL_MODEL)
    output_base = RESULTS_DIR / subdir
    output_base.mkdir(parents=True, exist_ok=True)

    pairs = []
    for ds in ID_DATASETS:
        pairs.append((ds, ds))
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src != tgt:
                pairs.append((src, tgt))

    if dry_run:
        print(f"  [DRY RUN] Would evaluate {len(pairs)} pairs:")
        for src, tgt in pairs:
            tag = "ID" if src == tgt else "OOD"
            print(f"    {src} -> {tgt} ({tag})")
        return

    for src, tgt in pairs:
        is_id = (src == tgt)
        if is_id:
            out_path = output_base / f'{src}_id.json'
        else:
            out_path = output_base / f'{src}_to_{tgt}_ood.json'

        if out_path.exists():
            logger.info(f"Skipping (exists): {out_path}")
            continue

        result = evaluate_pair(src, tgt)
        if result:
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved: {out_path}")


def description() -> str:
    return "RQ2.1: Evaluate SER-F1 for Matched/Unmatched/Balanced codebooks"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=description())
    parser.add_argument('--ssl-model', type=str, default=DEFAULT_SSL_MODEL,
                        choices=['e2v', 'hubert', 'wavlm'])
    parser.add_argument('--codebook-config', type=str, default=DEFAULT_CB_CONFIG,
                        help='Codebook config (e.g. 2x24, 1024x24)')
    parser.add_argument('--codebook-dataset', type=str, required=True,
                        help='Dataset whose codebooks to use (source)')
    parser.add_argument('--test-dataset', type=str, default=None,
                        help='Dataset to test on. If None, same as codebook-dataset (ID)')
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument('--num-layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing result files')
    args = parser.parse_args()

    if args.test_dataset is None:
        args.test_dataset = args.codebook_dataset

    if args.output_dir is None:
        args.output_dir = str(RESULTS_DIR / _result_subdir(args.codebook_config, args.ssl_model))

    result = evaluate_pair(
        codebook_dataset=args.codebook_dataset,
        test_dataset=args.test_dataset,
        ssl_model=args.ssl_model,
        cb_config=args.codebook_config,
        max_samples=args.max_samples,
        num_layers=args.num_layers,
        device=args.device,
    )

    if not result:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_id = (args.codebook_dataset == args.test_dataset)
    if is_id:
        out_path = output_dir / f'{args.codebook_dataset}_id.json'
    else:
        out_path = output_dir / f'{args.codebook_dataset}_to_{args.test_dataset}_ood.json'

    if out_path.exists() and not args.force:
        logger.info(f"Skipping (exists, use --force to overwrite): {out_path}")
        return

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
