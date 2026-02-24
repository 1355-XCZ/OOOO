#!/usr/bin/env python3
"""
RQ3.1 Sample-Level Evaluator -- Per-sample, per-codebook predictions

Saves detailed per-sample data for analysis:
  - For each ratio type and each emotion-specific codebook:
    true_label, predicted_label, cosine_sim, per-class F1

10 ratio types: balanced, 10+90, 20+80, 40+60, 50+50, 70+30, 80+20, 95+5, 99+1, biased

Output:
  results/rq3_1_ratio_samples/{cb_config}/{ssl}/{src}_to_{tgt}_ood_samples.json
  results/rq3_1_ratio_samples/{cb_config}/{ssl}/{src}_to_{tgt}_ood_agg.json

Usage:
  python -m paper_pipeline.evaluators.rq3_1_sample_level \
      --ssl-model e2v --codebook-config 2x24 \
      --codebook-dataset esd_en --test-dataset iemocap
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

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']
FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

RATIO_CODEBOOK_TYPES = [
    ('balanced',   'balanced.pt',            False),
    ('mixed_r10',  'mixed_{emo}_r10.pt',     True),
    ('mixed_r20',  'mixed_{emo}_r20.pt',     True),
    ('mixed_r40',  'mixed_{emo}_r40.pt',     True),
    ('mixed_r50',  'mixed_{emo}_r50.pt',     True),
    ('mixed_r70',  'mixed_{emo}_r70.pt',     True),
    ('mixed_r80',  'mixed_{emo}_r80.pt',     True),
    ('mixed_r95',  'mixed_{emo}_r95.pt',     True),
    ('mixed_r99',  'mixed_{emo}_r99.pt',     True),
    ('biased',     'biased_{emo}.pt',        True),
]


def _result_subdir(cb_config: str):
    return f'rq3_1_ratio_samples/{cb_config}'


def _predict_single(model, features, e2v_head, valid_indices, valid_e2v_labels, target_layer, device):
    """Get prediction and cosine for a single sample at target_layer."""
    is_enc_dec = isinstance(model, EncoderDecoderRVQ)
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

        for layer in range(1, target_layer + 1):
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
        probs_dict = {e: float(probs[i]) for i, e in enumerate(valid_e2v_labels)}

    return pred_label, cos_sim, probs_dict


def evaluate_pair(
    ssl_model: str,
    cb_config: str,
    codebook_dataset: str,
    test_dataset: str,
    max_samples: int = 200,
    num_layers: int = 24,
    device: str = 'cuda',
):
    target_layer = num_layers

    cb_ds_config = DATASET_CONFIGS[codebook_dataset]
    test_ds_config = DATASET_CONFIGS[test_dataset]
    cb_e2v_map = cb_ds_config.emotion_to_e2v
    test_e2v_map = test_ds_config.emotion_to_e2v

    cb_e2v_labels = {cb_e2v_map[e] for e in cb_ds_config.emotions if e in cb_e2v_map}
    test_e2v_labels = {test_e2v_map[e] for e in test_ds_config.emotions if e in test_e2v_map}
    common_e2v = sorted(cb_e2v_labels & test_e2v_labels & set(FAIR_EMOTIONS))

    if not common_e2v:
        logger.error(f"No overlapping fair emotions")
        return None, None

    e2v_to_cb_emotion = {cb_e2v_map[e]: e for e in cb_ds_config.emotions if cb_e2v_map.get(e) in common_e2v}
    e2v_to_test_emotion = {test_e2v_map[e]: e for e in test_ds_config.emotions if test_e2v_map.get(e) in common_e2v}

    logger.info(f"RQ3.1 Sample-Level [{ssl_model}]: {codebook_dataset} -> {test_dataset}")

    with open(SPLITS_DIR / test_dataset / 'test.json') as f:
        test_splits = json.load(f)

    cb_dir = get_codebook_dir(str(CODEBOOK_DIR), ssl_model, codebook_dataset, config=cb_config)

    if ssl_model == 'e2v':
        e2v_head = load_e2v_head(E2V_HEAD_PATH, device)
    else:
        e2v_head = load_custom_head(source=codebook_dataset, ssl_model=ssl_model, device=device)

    valid_indices = [E2V_LABELS.index(l) for l in common_e2v if l in E2V_LABELS]
    valid_e2v_labels = [l for l in common_e2v if l in E2V_LABELS]

    extractor = get_ssl_extractor(ssl_model, device=device)

    all_samples = []
    for e2v_label in common_e2v:
        test_emotion = e2v_to_test_emotion[e2v_label]
        files = test_splits.get(test_emotion, [])
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]
        for fpath in files:
            feats = extract_features(extractor, fpath)
            if feats is not None:
                all_samples.append({'features': feats, 'true_label': e2v_label, 'audio_path': fpath})

    del extractor
    torch.cuda.empty_cache()

    logger.info(f"  Loaded {len(all_samples)} samples, emotions: {common_e2v}")

    sample_records = []
    for i, s in enumerate(all_samples):
        sample_records.append({
            'idx': i,
            'true_label': s['true_label'],
            'audio_path': s['audio_path'],
        })

    agg_results = {}

    for ratio_key, fname_tpl, is_emo_specific in RATIO_CODEBOOK_TYPES:
        logger.info(f"  Evaluating: {ratio_key}")

        if not is_emo_specific:
            cb_path = cb_dir / fname_tpl
            codebook = load_codebook(str(cb_path), device)
            if not codebook:
                logger.warning(f"  Missing: {cb_path}")
                continue

            preds, coss = [], []
            for i, s in enumerate(tqdm(all_samples, desc=ratio_key, leave=False)):
                pred, cos, probs = _predict_single(
                    codebook, s['features'], e2v_head,
                    valid_indices, valid_e2v_labels, target_layer, device)
                sample_records[i][ratio_key] = {
                    'pred': pred, 'cosine': round(cos, 6), 'probs': {k: round(v, 4) for k, v in probs.items()}
                }
                preds.append(pred)
                coss.append(cos)

            trues = [s['true_label'] for s in all_samples]
            macro_f1 = float(f1_score(trues, preds, labels=common_e2v, average='macro', zero_division=0))
            per_class = f1_score(trues, preds, labels=common_e2v, average=None, zero_division=0)
            agg_results[ratio_key] = {
                'f1_macro': macro_f1,
                'cosine': float(np.mean(coss)),
                'per_emotion_f1': {e: float(f) for e, f in zip(common_e2v, per_class)},
            }

        else:
            per_cb_data = {}
            for cb_emo in common_e2v:
                cb_emotion_raw = e2v_to_cb_emotion.get(cb_emo, cb_emo)
                cb_path = cb_dir / fname_tpl.format(emo=cb_emotion_raw)
                codebook = load_codebook(str(cb_path), device)
                if not codebook:
                    logger.warning(f"  Missing: {cb_path}")
                    continue

                preds, coss = [], []
                for i, s in enumerate(tqdm(all_samples, desc=f"{ratio_key}_cb_{cb_emo}", leave=False)):
                    pred, cos, probs = _predict_single(
                        codebook, s['features'], e2v_head,
                        valid_indices, valid_e2v_labels, target_layer, device)
                    cb_key = f'{ratio_key}__cb_{cb_emo}'
                    sample_records[i][cb_key] = {
                        'pred': pred, 'cosine': round(cos, 6), 'probs': {k: round(v, 4) for k, v in probs.items()}
                    }
                    preds.append(pred)
                    coss.append(cos)

                trues = [s['true_label'] for s in all_samples]
                macro_f1 = float(f1_score(trues, preds, labels=common_e2v, average='macro', zero_division=0))
                per_class = f1_score(trues, preds, labels=common_e2v, average=None, zero_division=0)
                per_cb_data[cb_emo] = {
                    'f1_macro': macro_f1,
                    'cosine': float(np.mean(coss)),
                    'per_emotion_f1': {e: float(f) for e, f in zip(common_e2v, per_class)},
                }

            if per_cb_data:
                avg_f1 = float(np.mean([d['f1_macro'] for d in per_cb_data.values()]))
                avg_cos = float(np.mean([d['cosine'] for d in per_cb_data.values()]))
                avg_per_emo = {}
                for e in common_e2v:
                    avg_per_emo[e] = float(np.mean([d['per_emotion_f1'].get(e, 0) for d in per_cb_data.values()]))
                agg_results[ratio_key] = {
                    'f1_macro': avg_f1,
                    'cosine': avg_cos,
                    'per_emotion_f1': avg_per_emo,
                    'per_codebook': per_cb_data,
                }

    return sample_records, agg_results


def main():
    parser = argparse.ArgumentParser(description='RQ3.1 Sample-Level Evaluator')
    parser.add_argument('--ssl-model', type=str, required=True, choices=['e2v', 'hubert', 'wavlm'])
    parser.add_argument('--codebook-config', type=str, default='2x24')
    parser.add_argument('--codebook-dataset', type=str, required=True)
    parser.add_argument('--test-dataset', type=str, required=True)
    parser.add_argument('--num-layers', type=int, default=24)
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    sample_records, agg_results = evaluate_pair(
        args.ssl_model, args.codebook_config,
        args.codebook_dataset, args.test_dataset,
        max_samples=args.max_samples, num_layers=args.num_layers,
        device=args.device,
    )

    if sample_records is None:
        return

    is_id = (args.codebook_dataset == args.test_dataset)
    suffix = f'{args.codebook_dataset}_id' if is_id else f'{args.codebook_dataset}_to_{args.test_dataset}_ood'

    out_dir = RESULTS_DIR / _result_subdir(args.codebook_config) / args.ssl_model
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / f'{suffix}_samples.json'
    with open(samples_path, 'w') as f:
        json.dump(sample_records, f, indent=1)
    logger.info(f"  Saved samples: {samples_path}")

    agg_path = out_dir / f'{suffix}_agg.json'
    with open(agg_path, 'w') as f:
        json.dump(agg_results, f, indent=2)
    logger.info(f"  Saved aggregate: {agg_path}")

    print(f"\n{'='*80}")
    print(f"  {args.codebook_dataset} -> {args.test_dataset} [{args.ssl_model}]")
    print(f"{'='*80}")
    for rk, data in agg_results.items():
        f1 = data['f1_macro']
        cos = data['cosine']
        per_emo = data.get('per_emotion_f1', {})
        emo_str = " | ".join(f"{e}:{v:.4f}" for e, v in per_emo.items())
        print(f"  {rk:>12s}: F1={f1:.4f} cos={cos:.4f} | {emo_str}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
