#!/usr/bin/env python3
"""
RQ2.2 Evaluator -- L2 + SER-F1 for Matched/Unmatched/Balanced across SSL models

Generalizes the RQ2.1 evaluator to support e2v, hubert, and wavlm,
computing both L2 distance and SER-F1 per layer.

Classification heads:
  - e2v:   pre-trained E2V head (dataset-agnostic)
  - hubert/wavlm: dataset-specific LinearProbe (from codebook source)

Output per (ssl, codebook_dataset, test_dataset):
  results/rq2_2_ssl_table_128x8/{ssl}/{dataset}_id.json
  results/rq2_2_ssl_table_128x8/{ssl}/{src}_to_{tgt}_ood.json

Standalone:
  python -m paper_pipeline.evaluators.rq2_2_ssl_table \
      --ssl-model hubert --codebook-dataset esd_en --test-dataset iemocap
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
from sklearn.metrics import accuracy_score, f1_score

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import DEFAULT_MAX_SAMPLES
from configs.dataset_config import DATASET_CONFIGS
from core.config import (
    E2V_LABELS, E2V_HEAD_PATH, CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR,
)
from core.features import get_ssl_extractor, extract_features
from core.quantize import load_codebook, compute_similarity, EncoderDecoderRVQ
from core.classify import load_e2v_head, load_custom_head

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RESULT_SUBDIR = 'rq2_2_ssl_table_128x8'
ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']
NUM_LAYERS = 8
CB_CONFIG = '128x8'
SSL_MODELS = ['e2v', 'hubert', 'wavlm']


def pre_extract_features(extractor, samples):
    cached = []
    for audio_path, label in tqdm(samples, desc="Pre-extracting features"):
        feats = extract_features(extractor, audio_path)
        if feats is not None:
            cached.append((feats, label))
    logger.info(f"Cached {len(cached)}/{len(samples)} samples")
    return cached


def evaluate_codebook_on_samples(
    model, cached_samples, e2v_head,
    valid_indices, valid_e2v_labels,
    num_layers, device, desc="",
):
    """Returns {layer: [(true_label, pred_label, l2_dist), ...]}."""
    is_enc_dec = isinstance(model, EncoderDecoderRVQ)
    layer_results: Dict[int, list] = defaultdict(list)

    for features, true_label in tqdm(cached_samples, desc=desc, leave=False):
        orig = features.to(device)
        feat_input = orig.unsqueeze(0) if orig.dim() == 2 else orig

        with torch.no_grad():
            if is_enc_dec:
                encoded = model.encoder(feat_input)
                _, indices, _ = model.rvq(encoded)
                partial = torch.zeros_like(encoded)
            else:
                _, indices, _, _ = model(feat_input)
                partial = torch.zeros_like(feat_input)

            for layer in range(1, num_layers + 1):
                layer_idx = indices[:, :, layer - 1]
                cb_weights = model.rvq.layers[layer - 1]._codebook.embed[0]
                partial = partial + cb_weights[layer_idx]

                if is_enc_dec:
                    quantized = model.decoder(partial).squeeze(0)
                else:
                    quantized = partial.squeeze(0)

                l2_dist = compute_similarity(orig, quantized, metric='l2')

                q_mean = quantized.mean(dim=0).unsqueeze(0)
                logits = e2v_head(q_mean)
                valid_logits = logits[:, valid_indices]
                probs = F.softmax(valid_logits, dim=-1).squeeze(0)
                pred_label = valid_e2v_labels[probs.argmax().item()]

                layer_results[layer].append((true_label, pred_label, l2_dist))

    return dict(layer_results)


def compute_metrics(results_list):
    if not results_list:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'l2': 0.0, 'num_samples': 0}
    y_true = [r[0] for r in results_list]
    y_pred = [r[1] for r in results_list]
    l2_vals = [abs(r[2]) for r in results_list]
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'l2': float(np.mean(l2_vals)),
        'num_samples': len(results_list),
    }


def evaluate_pair(
    ssl_model: str,
    codebook_dataset: str,
    test_dataset: str,
    max_samples: int = 200,
    num_layers: int = NUM_LAYERS,
    device: str = 'cuda',
):
    codebook_base_dir = CODEBOOK_DIR / ssl_model / CB_CONFIG
    is_id = (codebook_dataset == test_dataset)
    eval_type = "ID" if is_id else "OOD"

    cb_config = DATASET_CONFIGS[codebook_dataset]
    test_config = DATASET_CONFIGS[test_dataset]

    cb_e2v_map = cb_config.emotion_to_e2v
    test_e2v_map = test_config.emotion_to_e2v

    cb_e2v_labels = {cb_e2v_map[e] for e in cb_config.emotions if e in cb_e2v_map}
    test_e2v_labels = {test_e2v_map[e] for e in test_config.emotions if e in test_e2v_map}
    common_e2v = sorted(cb_e2v_labels & test_e2v_labels)

    if not common_e2v:
        logger.error(f"No overlapping emotions: {codebook_dataset} vs {test_dataset}")
        return {}

    e2v_to_cb_emotion = {cb_e2v_map[e]: e for e in cb_config.emotions if cb_e2v_map.get(e) in common_e2v}
    e2v_to_test_emotion = {test_e2v_map[e]: e for e in test_config.emotions if test_e2v_map.get(e) in common_e2v}

    logger.info(f"RQ2.2 [{ssl_model}] ({eval_type}): {codebook_dataset} -> {test_dataset}")

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

    # Load appropriate classification head
    if ssl_model == 'e2v':
        e2v_head = load_e2v_head(E2V_HEAD_PATH, device)
    else:
        e2v_head = load_custom_head(source=codebook_dataset, ssl_model=ssl_model, device=device)

    valid_indices = [E2V_LABELS.index(l) for l in common_e2v if l in E2V_LABELS]
    valid_e2v_labels = [l for l in common_e2v if l in E2V_LABELS]

    extractor = get_ssl_extractor(ssl_model, device=device)
    cached_per_emotion = {}
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

    eval_emotions = sorted(set(cached_per_emotion.keys()) & set(biased_codebooks.keys()))
    logger.info(f"  Eval emotions: {eval_emotions}")

    filtered_samples = []
    for emo in eval_emotions:
        filtered_samples.extend(cached_per_emotion[emo])

    results = {}

    # --- Balanced ---
    if balanced_codebook:
        layer_res = evaluate_codebook_on_samples(
            balanced_codebook, filtered_samples, e2v_head,
            valid_indices, valid_e2v_labels, num_layers, device, "Balanced")
        results['balanced'] = {
            f'layer_{l}': compute_metrics(layer_res.get(l, []))
            for l in range(1, num_layers + 1)
        }

    # --- Matched ---
    matched_layer_res = defaultdict(list)
    for emo in eval_emotions:
        layer_res = evaluate_codebook_on_samples(
            biased_codebooks[emo], cached_per_emotion[emo], e2v_head,
            valid_indices, valid_e2v_labels, num_layers, device, f"Matched-{emo}")
        for l, pairs in layer_res.items():
            matched_layer_res[l].extend(pairs)
    results['biased_matched'] = {
        f'layer_{l}': compute_metrics(matched_layer_res.get(l, []))
        for l in range(1, num_layers + 1)
    }

    # --- Unmatched ---
    unmatched_layer_res = defaultdict(list)
    for emo in eval_emotions:
        for other in eval_emotions:
            if other == emo:
                continue
            layer_res = evaluate_codebook_on_samples(
                biased_codebooks[other], cached_per_emotion[emo], e2v_head,
                valid_indices, valid_e2v_labels, num_layers, device,
                f"Unmatched-{other}-on-{emo}")
            for l, pairs in layer_res.items():
                unmatched_layer_res[l].extend(pairs)
    results['biased_unmatched'] = {
        f'layer_{l}': compute_metrics(unmatched_layer_res.get(l, []))
        for l in range(1, num_layers + 1)
    }

    return {
        'config': {
            'ssl_model': ssl_model,
            'codebook_dataset': codebook_dataset,
            'test_dataset': test_dataset,
            'eval_type': eval_type,
            'num_layers': num_layers,
            'eval_emotions': eval_emotions,
        },
        **results,
    }


def run(dry_run=False):
    output_base = RESULTS_DIR / RESULT_SUBDIR

    pairs = []
    for ds in ID_DATASETS:
        pairs.append((ds, ds))
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src != tgt:
                pairs.append((src, tgt))

    if dry_run:
        total = len(SSL_MODELS) * len(pairs)
        print(f"  [DRY RUN] Would evaluate {total} tasks ({len(SSL_MODELS)} SSLs x {len(pairs)} pairs)")
        return

    for ssl in SSL_MODELS:
        for src, tgt in pairs:
            ssl_dir = output_base / ssl
            ssl_dir.mkdir(parents=True, exist_ok=True)
            is_id = (src == tgt)
            out_path = ssl_dir / (f'{src}_id.json' if is_id else f'{src}_to_{tgt}_ood.json')

            if out_path.exists():
                logger.info(f"Skipping (exists): {out_path}")
                continue

            result = evaluate_pair(ssl, src, tgt)
            if result:
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved: {out_path}")


def description():
    return "RQ2.2: Evaluate L2 + SER-F1 for all SSL models (128x8)"


def main():
    parser = argparse.ArgumentParser(description=description())
    parser.add_argument('--ssl-model', type=str, required=True, choices=SSL_MODELS)
    parser.add_argument('--codebook-dataset', type=str, required=True)
    parser.add_argument('--test-dataset', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR / RESULT_SUBDIR))
    args = parser.parse_args()

    if args.test_dataset is None:
        args.test_dataset = args.codebook_dataset

    result = evaluate_pair(args.ssl_model, args.codebook_dataset, args.test_dataset,
                           max_samples=args.max_samples, device=args.device)
    if not result:
        return

    output_dir = Path(args.output_dir) / args.ssl_model
    output_dir.mkdir(parents=True, exist_ok=True)
    is_id = (args.codebook_dataset == args.test_dataset)
    out_path = output_dir / (f'{args.codebook_dataset}_id.json' if is_id
                             else f'{args.codebook_dataset}_to_{args.test_dataset}_ood.json')

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
