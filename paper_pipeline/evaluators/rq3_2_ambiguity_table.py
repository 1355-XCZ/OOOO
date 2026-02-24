#!/usr/bin/env python3
"""
RQ3.2 Evaluator -- Cosine + SER-F1 for ambiguity codebooks across SSL models

Evaluates 3 ambiguity levels (high/mid/low) per (ssl, test_dataset):
  Codebook source is always IEMOCAP.
  For each emotion, the matched ambiguity codebook is used.

Metrics: cosine similarity + SER F1-Macro, aggregated across emotions.

Classification heads:
  - e2v:   pre-trained E2V head
  - hubert/wavlm: dataset-specific LinearProbe (from iemocap)

Output per (ssl, test_dataset):
  results/rq3_2_ambiguity/{cb_config}/{ssl}/iemocap_to_{tgt}_ood.json

Standalone:
  python -m paper_pipeline.evaluators.rq3_2_ambiguity_table \
      --ssl-model e2v --codebook-config 2x24 --test-dataset esd_en
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import DEFAULT_MAX_SAMPLES
from configs.dataset_config import DATASET_CONFIGS
from core.config import (
    E2V_LABELS, E2V_HEAD_PATH, CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR,
)
from core.features import get_ssl_extractor, extract_features, get_codebook_dir
from core.quantize import load_codebook, compute_similarity, EncoderDecoderRVQ
from core.classify import load_e2v_head, load_custom_head

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_SSL_MODEL = 'e2v'
DEFAULT_CB_CONFIG = '2x24'
DEFAULT_NUM_LAYERS = 24
CODEBOOK_DATASET = 'iemocap'
FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
AMBIGUITY_LEVELS = ['high', 'mid', 'low']


def _result_subdir(cb_config: str):
    return f'rq3_2_ambiguity/{cb_config}'


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

                cos_sim = compute_similarity(orig, quantized, metric='cosine')

                q_mean = quantized.mean(dim=0).unsqueeze(0)
                logits = e2v_head(q_mean)
                valid_logits = logits[:, valid_indices]
                probs = F.softmax(valid_logits, dim=-1).squeeze(0)
                pred_label = valid_e2v_labels[probs.argmax().item()]

                layer_results[layer].append((true_label, pred_label, cos_sim))

    return dict(layer_results)


def compute_metrics(results_list):
    if not results_list:
        return {'f1_macro': 0.0, 'cosine': 0.0, 'num_samples': 0}
    y_true = [r[0] for r in results_list]
    y_pred = [r[1] for r in results_list]
    cos_vals = [r[2] for r in results_list]
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'cosine': float(np.mean(cos_vals)),
        'num_samples': len(results_list),
    }


def evaluate_pair(
    ssl_model: str,
    cb_config: str,
    test_dataset: str,
    max_samples: int = 200,
    num_layers: int = DEFAULT_NUM_LAYERS,
    device: str = 'cuda',
):
    is_id = (CODEBOOK_DATASET == test_dataset)
    eval_type = "ID" if is_id else "OOD"

    cb_ds_config = DATASET_CONFIGS[CODEBOOK_DATASET]
    test_ds_config = DATASET_CONFIGS[test_dataset]

    cb_e2v_map = cb_ds_config.emotion_to_e2v
    test_e2v_map = test_ds_config.emotion_to_e2v

    cb_e2v_labels = {cb_e2v_map[e] for e in cb_ds_config.emotions if e in cb_e2v_map}
    test_e2v_labels = {test_e2v_map[e] for e in test_ds_config.emotions if e in test_e2v_map}
    common_e2v = sorted(cb_e2v_labels & test_e2v_labels & set(FAIR_EMOTIONS))

    if not common_e2v:
        logger.error(f"No overlapping fair emotions: {CODEBOOK_DATASET} vs {test_dataset}")
        return {}

    e2v_to_cb_emotion = {cb_e2v_map[e]: e for e in cb_ds_config.emotions if cb_e2v_map.get(e) in common_e2v}
    e2v_to_test_emotion = {test_e2v_map[e]: e for e in test_ds_config.emotions if test_e2v_map.get(e) in common_e2v}

    logger.info(f"RQ3.2 [{ssl_model}] ({eval_type}): {CODEBOOK_DATASET} -> {test_dataset}")

    with open(SPLITS_DIR / test_dataset / 'test.json') as f:
        test_splits = json.load(f)

    cb_dir = get_codebook_dir(str(CODEBOOK_DIR), ssl_model, CODEBOOK_DATASET, config=cb_config)

    if ssl_model == 'e2v':
        e2v_head = load_e2v_head(E2V_HEAD_PATH, device)
    else:
        e2v_head = load_custom_head(source=CODEBOOK_DATASET, ssl_model=ssl_model, device=device)

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

    eval_emotions = sorted(set(cached_per_emotion.keys()))
    logger.info(f"  Eval emotions: {eval_emotions}")

    results = {}

    all_samples = []
    for emo in eval_emotions:
        all_samples.extend(cached_per_emotion[emo])

    for level in AMBIGUITY_LEVELS:
        per_cb_metrics = []
        for e2v_label in eval_emotions:
            cb_emotion = e2v_to_cb_emotion.get(e2v_label, e2v_label)
            cb_path = cb_dir / f'ambiguity_{cb_emotion}_{level}.pt'
            codebook = load_codebook(str(cb_path), device)
            if not codebook:
                logger.warning(f"  Missing codebook: {cb_path}")
                continue

            layer_res = evaluate_codebook_on_samples(
                codebook, all_samples, e2v_head,
                valid_indices, valid_e2v_labels, num_layers, device,
                f"Ambiguity-{level}-cb_{e2v_label}")
            per_cb_metrics.append({
                l: compute_metrics(layer_res.get(l, []))
                for l in range(1, num_layers + 1)
            })

        if per_cb_metrics:
            avg_result = {}
            for l in range(1, num_layers + 1):
                f1s = [m[l]['f1_macro'] for m in per_cb_metrics if l in m]
                coss = [m[l]['cosine'] for m in per_cb_metrics if l in m]
                ns = [m[l]['num_samples'] for m in per_cb_metrics if l in m]
                avg_result[f'layer_{l}'] = {
                    'f1_macro': float(np.mean(f1s)) if f1s else 0.0,
                    'cosine': float(np.mean(coss)) if coss else 0.0,
                    'num_samples': int(np.mean(ns)) if ns else 0,
                }
            results[level] = avg_result

    return {
        'config': {
            'ssl_model': ssl_model,
            'cb_config': cb_config,
            'codebook_dataset': CODEBOOK_DATASET,
            'test_dataset': test_dataset,
            'eval_type': eval_type,
            'num_layers': num_layers,
            'eval_emotions': eval_emotions,
        },
        **results,
    }


def description():
    return "RQ3.2: Ambiguity codebook evaluation (Cosine + SER-F1, IEMOCAP)"


def main():
    parser = argparse.ArgumentParser(description=description())
    parser.add_argument('--ssl-model', type=str, required=True,
                        choices=['e2v', 'hubert', 'wavlm'])
    parser.add_argument('--codebook-config', type=str, default=DEFAULT_CB_CONFIG,
                        help='Codebook config (e.g. 2x24, 1024x24)')
    parser.add_argument('--test-dataset', type=str, required=True)
    parser.add_argument('--num-layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    result = evaluate_pair(
        args.ssl_model, args.codebook_config,
        args.test_dataset,
        max_samples=args.max_samples, num_layers=args.num_layers,
        device=args.device,
    )
    if not result:
        return

    subdir = _result_subdir(args.codebook_config)
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / subdir / args.ssl_model
    output_dir.mkdir(parents=True, exist_ok=True)

    is_id = (CODEBOOK_DATASET == args.test_dataset)
    out_path = output_dir / (f'{CODEBOOK_DATASET}_id.json' if is_id
                             else f'{CODEBOOK_DATASET}_to_{args.test_dataset}_ood.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
