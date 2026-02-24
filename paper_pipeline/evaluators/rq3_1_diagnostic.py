#!/usr/bin/env python3
"""
RQ3.1 Diagnostic -- Per-codebook, per-emotion F1 breakdown

Shows which emotion benefits/suffers from each ratio codebook.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.dataset_config import DATASET_CONFIGS
from core.config import E2V_LABELS, E2V_HEAD_PATH, CODEBOOK_DIR, SPLITS_DIR
from core.features import get_ssl_extractor, extract_features, get_codebook_dir
from core.quantize import load_codebook, get_all_reconstructions
from core.classify import load_e2v_head

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
CB_CONFIG = '2x24'
NUM_LAYERS = 24
TARGET_LAYER = 24


def evaluate_detailed(codebook_dataset, test_dataset, device='cuda'):
    cb_ds_config = DATASET_CONFIGS[codebook_dataset]
    test_ds_config = DATASET_CONFIGS[test_dataset]
    cb_e2v_map = cb_ds_config.emotion_to_e2v
    test_e2v_map = test_ds_config.emotion_to_e2v

    common_e2v = sorted(set(FAIR_EMOTIONS))
    e2v_to_cb_emotion = {cb_e2v_map[e]: e for e in cb_ds_config.emotions if cb_e2v_map.get(e) in common_e2v}
    e2v_to_test_emotion = {test_e2v_map[e]: e for e in test_ds_config.emotions if test_e2v_map.get(e) in common_e2v}

    with open(SPLITS_DIR / test_dataset / 'test.json') as f:
        test_splits = json.load(f)

    cb_dir = get_codebook_dir(str(CODEBOOK_DIR), 'e2v', codebook_dataset, config=CB_CONFIG)
    e2v_head = load_e2v_head(E2V_HEAD_PATH, device)
    valid_indices = [E2V_LABELS.index(l) for l in common_e2v if l in E2V_LABELS]

    extractor = get_ssl_extractor('e2v', device=device)

    all_samples = []
    for e2v_label in common_e2v:
        test_emotion = e2v_to_test_emotion.get(e2v_label)
        if not test_emotion:
            continue
        files = test_splits.get(test_emotion, [])[:200]
        for fpath in files:
            feats = extract_features(extractor, fpath)
            if feats is not None:
                all_samples.append((feats, e2v_label))

    del extractor
    torch.cuda.empty_cache()

    logger.info(f"Loaded {len(all_samples)} samples, emotions: {common_e2v}")

    trues = [s[1] for s in all_samples]

    codebook_types = [
        ('balanced', 'balanced.pt', False),
        ('mixed_r80', 'mixed_{emo}_r80.pt', True),
        ('mixed_r95', 'mixed_{emo}_r95.pt', True),
        ('biased', 'biased_{emo}.pt', True),
    ]

    print(f"\n{'='*100}")
    print(f"  {codebook_dataset} -> {test_dataset} | Layer {TARGET_LAYER}")
    print(f"  Samples: {len(all_samples)}")
    print(f"{'='*100}")

    for ratio_key, fname_tpl, is_emo_specific in codebook_types:
        print(f"\n--- {ratio_key.upper()} ---")

        if not is_emo_specific:
            cb = load_codebook(str(cb_dir / fname_tpl), device)
            preds = _predict_all(cb, all_samples, e2v_head, valid_indices, common_e2v, device)
            _print_report(trues, preds, common_e2v, f"  {ratio_key}")
        else:
            all_cb_preds = {}
            for cb_emo in common_e2v:
                cb_emotion_raw = e2v_to_cb_emotion.get(cb_emo, cb_emo)
                cb_path = cb_dir / fname_tpl.format(emo=cb_emotion_raw)
                cb = load_codebook(str(cb_path), device)
                if not cb:
                    continue
                preds = _predict_all(cb, all_samples, e2v_head, valid_indices, common_e2v, device)
                all_cb_preds[cb_emo] = preds

                per_emo_f1 = {}
                for emo in common_e2v:
                    emo_idx = [i for i, t in enumerate(trues) if t == emo]
                    emo_trues = [trues[i] for i in emo_idx]
                    emo_preds = [preds[i] for i in emo_idx]
                    if emo_trues:
                        acc = np.mean([t == p for t, p in zip(emo_trues, emo_preds)])
                        per_emo_f1[emo] = acc
                    else:
                        per_emo_f1[emo] = 0

                macro_f1 = f1_score(trues, preds, labels=common_e2v, average='macro', zero_division=0)
                per_class = f1_score(trues, preds, labels=common_e2v, average=None, zero_division=0)
                detail = " | ".join(f"{e}: F1={f:.4f}" for e, f in zip(common_e2v, per_class))
                print(f"  cb_{cb_emo:>8s}: F1-macro={macro_f1:.4f} | {detail}")

            if all_cb_preds:
                avg_f1s = []
                for cb_emo, preds in all_cb_preds.items():
                    avg_f1s.append(f1_score(trues, preds, labels=common_e2v, average='macro', zero_division=0))

                per_class_avg = np.zeros(len(common_e2v))
                for cb_emo, preds in all_cb_preds.items():
                    per_class_avg += f1_score(trues, preds, labels=common_e2v, average=None, zero_division=0)
                per_class_avg /= len(all_cb_preds)

                detail = " | ".join(f"{e}: F1={f:.4f}" for e, f in zip(common_e2v, per_class_avg))
                print(f"  {'AVG':>11s}: F1-macro={np.mean(avg_f1s):.4f} | {detail}")


def _predict_all(codebook, samples, e2v_head, valid_indices, labels, device):
    preds = []
    sl = TARGET_LAYER
    for feats, _ in samples:
        feats_gpu = feats.to(device)
        recons = get_all_reconstructions(codebook, feats_gpu, [sl], device)
        if sl not in recons:
            preds.append(labels[0])
            continue
        with torch.no_grad():
            logits = e2v_head(recons[sl].unsqueeze(0) if recons[sl].dim() == 2 else recons[sl])
            valid_logits = logits[:, valid_indices]
            pred_idx = valid_logits.argmax(dim=-1).item()
        preds.append(labels[pred_idx])
    return preds


def _print_report(trues, preds, labels, prefix):
    macro_f1 = f1_score(trues, preds, labels=labels, average='macro', zero_division=0)
    per_class = f1_score(trues, preds, labels=labels, average=None, zero_division=0)
    detail = " | ".join(f"{e}: F1={f:.4f}" for e, f in zip(labels, per_class))
    print(f"{prefix}: F1-macro={macro_f1:.4f} | {detail}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='esd_en')
    parser.add_argument('--tgt', default='iemocap')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    evaluate_detailed(args.src, args.tgt, args.device)
