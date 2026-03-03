#!/usr/bin/env python3
"""
Layer-wise Evaluation of Balanced 2x32 Codebook -- Per-Emotion, Per-Sample

Supports all SSL models (e2v, wavlm, hubert).
Stores SAMPLE-LEVEL results for maximum flexibility in downstream plotting.

For each (SSL, codebook_dataset, test_dataset) combination:
  - Extract features using the SSL model
  - Quantize with the corresponding 2x32 balanced codebook (L1..L32)
  - Compute per-sample cosine similarity
  - Classify with the corresponding SER classifier
  - Store sample-level data (true_label, cosines, preds)

Output JSON structure (per-sample):
  {
    "config": { ... },
    "samples": [
      {
        "true_label": "angry",          # canonical (e2v-mapped) label
        "cosines": [0.73, 0.81, ...],   # layer 1..32
        "preds": ["angry", "angry", ...]  # predicted label at each layer
      },
      ...
    ]
  }

Usage:
  python evaluate_ssl_balanced_2x32.py --ssl-model wavlm --codebook-dataset cremad
  python evaluate_ssl_balanced_2x32.py --ssl-model hubert --codebook-dataset esd --test-dataset cremad
  python evaluate_ssl_balanced_2x32.py --ssl-model e2v --codebook-dataset cremad --max-samples 200
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import DEFAULT_MAX_SAMPLES
from configs.dataset_config import DATASET_CONFIGS
from core.config import E2V_LABELS, E2V_HEAD_PATH, CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR, CLASSIFIER_DIR
from core.features import get_ssl_extractor, extract_features
from core.quantize import load_codebook, compute_similarity, get_all_reconstructions
from core.classify import E2VClassificationHead, load_e2v_head

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Linear probe (wavlm/hubert) -- script-specific
# ============================================================

class LinearProbeHead(nn.Module):
    """Linear probe classifier (dataset-specific)."""
    def __init__(self, weight, bias, emotions):
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0])
        self.proj.weight.data = weight
        self.proj.bias.data = bias
        self.emotions = emotions  # ordered list of emotion labels

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.proj(x)


def load_linear_probe(classifier_path: str, device: str = 'cuda') -> LinearProbeHead:
    """Load a trained linear probe classifier."""
    ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    emotions = ckpt['emotions']
    weight = state_dict['classifier.weight']
    bias = state_dict['classifier.bias']
    head = LinearProbeHead(weight, bias, emotions)
    head = head.to(device).eval()
    logger.info(f"  Loaded linear probe: {len(emotions)} classes, dim={weight.shape[1]}")
    logger.info(f"  Emotions: {emotions}")
    logger.info(f"  Val acc: {ckpt.get('val_acc', 'N/A')}")
    return head


# ============================================================
# Emotion intersection for OOD
# ============================================================

def get_emotion_intersection(
    cb_emotions: List[str], cb_e2v_map: Dict[str, str],
    test_emotions: List[str], test_e2v_map: Dict[str, str]
) -> Tuple[Set[str], List[str], List[str]]:
    """Get emotion intersection based on E2V labels."""
    cb_e2v_labels = {cb_e2v_map[e] for e in cb_emotions if e in cb_e2v_map}
    test_e2v_labels = {test_e2v_map[e] for e in test_emotions if e in test_e2v_map}
    common_e2v = cb_e2v_labels & test_e2v_labels

    cb_emotions_to_use = [e for e in cb_emotions if cb_e2v_map.get(e) in common_e2v]
    test_emotions_to_use = [e for e in test_emotions if test_e2v_map.get(e) in common_e2v]
    return common_e2v, cb_emotions_to_use, test_emotions_to_use


# ============================================================
# Feature pre-extraction
# ============================================================

def pre_extract_by_emotion(
    extractor,
    samples_by_emotion: Dict[str, List[str]],
    emotion_to_e2v: Dict[str, str],
    common_e2v: Set[str],
    max_samples: Optional[int] = None,
) -> Dict[str, List[Tuple[torch.Tensor, str]]]:
    """Pre-extract features grouped by emotion."""
    cached = {}
    for emotion, files in samples_by_emotion.items():
        e2v_label = emotion_to_e2v.get(emotion)
        if e2v_label is None or e2v_label not in common_e2v:
            continue
        if max_samples and len(files) > max_samples:
            files = files[:max_samples]
        items = []
        for path in tqdm(files, desc=f"Extract {emotion}->{e2v_label}"):
            feats = extract_features(extractor, path)
            if feats is not None:
                items.append((feats, e2v_label))
        if items:
            cached[e2v_label] = items
            logger.info(f"  {emotion} (->{e2v_label}): {len(items)}/{len(files)}")
    return cached


# ============================================================
# Classifier prediction helpers
# ============================================================

def classify_e2v(
    quantized_mean: torch.Tensor,
    e2v_head: E2VClassificationHead,
    valid_indices: List[int],
    valid_labels: List[str],
    device: str,
) -> str:
    """Classify with emotion2vec head (restrict to valid classes)."""
    x = quantized_mean.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = e2v_head(x)
        valid_logits = logits[:, valid_indices]
        pred_idx = F.softmax(valid_logits, dim=-1).argmax(dim=-1).item()
    return valid_labels[pred_idx]


def classify_linear_probe(
    quantized_mean: torch.Tensor,
    head: LinearProbeHead,
    valid_indices: List[int],
    valid_labels: List[str],
    device: str,
) -> str:
    """Classify with linear probe (restrict to valid classes)."""
    x = quantized_mean.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = head(x)
        valid_logits = logits[:, valid_indices]
        pred_idx = F.softmax(valid_logits, dim=-1).argmax(dim=-1).item()
    return valid_labels[pred_idx]


# ============================================================
# Main evaluation
# ============================================================

def evaluate_per_sample(
    model: nn.Module,
    cached_by_emotion: Dict[str, List[Tuple[torch.Tensor, str]]],
    classifier,  # E2VClassificationHead or LinearProbeHead
    classify_fn,  # classify_e2v or classify_linear_probe
    valid_indices: List[int],
    valid_labels: List[str],
    num_layers: int,
    device: str = 'cuda',
    metric: str = 'l2',
) -> List[Dict]:
    """
    Evaluate at all layers, storing per-sample results.

    Returns a list of sample dicts:
      [
        {
          "true_label": "angry",
          "cosines": [cos_L1, cos_L2, ..., cos_L32],
          "preds": [pred_L1, pred_L2, ..., pred_L32]
        },
        ...
      ]
    """
    # Flatten samples
    all_samples = []
    for emo, items in cached_by_emotion.items():
        for feat, label in items:
            all_samples.append((feat, label))

    logger.info(f"Evaluating {len(all_samples)} samples across {num_layers} layers...")

    layers = list(range(1, num_layers + 1))
    results = []
    for i, (features, true_label) in enumerate(tqdm(all_samples, desc="Evaluating")):
        sample = {
            "true_label": true_label,
            "cosines": [],
            "preds": [],
        }
        reconstructions = get_all_reconstructions(model, features, layers, device)
        for layer in layers:
            quantized = reconstructions[layer]
            cos = compute_similarity(features.to(device), quantized, metric)
            q_mean = quantized.mean(dim=0)
            pred = classify_fn(q_mean, classifier, valid_indices, valid_labels, device)
            sample["cosines"].append(cos)
            sample["preds"].append(pred)

        results.append(sample)

        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(all_samples)} samples")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Layer-wise SSL balanced codebook evaluation (sample-level output)'
    )
    parser.add_argument('--ssl-model', type=str, required=True,
                        choices=['e2v', 'wavlm', 'hubert'],
                        help='SSL model to use')
    parser.add_argument('--codebook-dataset', type=str, required=True,
                        help='Dataset whose codebook to use')
    parser.add_argument('--test-dataset', type=str, default=None,
                        help='Dataset to test on (default: same as codebook-dataset -> ID)')
    parser.add_argument('--codebook-dir', type=str, default=str(CODEBOOK_DIR))
    parser.add_argument('--codebook-config', type=str, default='2x32',
                        help='Codebook config dimension (e.g. 2x32), inserted in path as codebooks/{ssl}/{config}/{dataset}/')
    parser.add_argument('--classifier-dir', type=str, default=str(CLASSIFIER_DIR))
    parser.add_argument('--splits-dir', type=str, default=str(SPLITS_DIR))
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'ssl_comparison'))
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument('--num-layers', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--metric', type=str, default='l2',
                        choices=['cosine', 'l2'],
                        help='Similarity metric: cosine or l2 (negative L2 distance)')
    parser.add_argument('--emotions', type=str, default=None,
                        help='Comma-separated E2V emotion subset, e.g. angry,happy,neutral,sad. If set, only these are evaluated.')
    args = parser.parse_args()

    if args.test_dataset is None:
        args.test_dataset = args.codebook_dataset
    is_id = (args.codebook_dataset == args.test_dataset)
    eval_type = "ID" if is_id else "OOD"

    cb_config = DATASET_CONFIGS[args.codebook_dataset]
    test_config = DATASET_CONFIGS[args.test_dataset]
    ssl_model = args.ssl_model

    logger.info("=" * 60)
    logger.info(f"SSL Balanced Codebook Evaluation ({eval_type})")
    logger.info(f"  SSL model:       {ssl_model}")
    logger.info(f"  Codebook source: {args.codebook_dataset}")
    logger.info(f"  Test target:     {args.test_dataset}")
    logger.info(f"  Num layers:      {args.num_layers}")
    logger.info(f"  Max samples/emo: {args.max_samples}")
    logger.info("=" * 60)

    device = args.device if torch.cuda.is_available() else 'cpu'

    # ---- Determine codebook path ----
    # Unified convention: codebooks/{ssl_model}/{config}/{dataset}/
    ckpt_name = 'balanced.pt'
    codebook_base = Path(args.codebook_dir)
    if ssl_model == 'e2v':
        codebook_path = codebook_base / 'e2v' / args.codebook_config / args.codebook_dataset / ckpt_name
    else:
        codebook_path = codebook_base / ssl_model / args.codebook_config / args.codebook_dataset / ckpt_name

    if not codebook_path.exists():
        logger.error(f"Codebook not found: {codebook_path}")
        return
    model = load_codebook(str(codebook_path), device)
    logger.info(f" Loaded codebook: {codebook_path}")

    # ---- Emotion intersection ----
    if is_id:
        common_e2v = set(cb_config.emotion_to_e2v.values())
        test_emotions_to_use = test_config.emotions
        eval_emotion_to_e2v = test_config.emotion_to_e2v
    else:
        common_e2v, _, test_emotions_to_use = get_emotion_intersection(
            cb_config.emotions, cb_config.emotion_to_e2v,
            test_config.emotions, test_config.emotion_to_e2v
        )
        eval_emotion_to_e2v = {e: test_config.emotion_to_e2v[e] for e in test_emotions_to_use}

    # ---- Optional: restrict to subset of emotions (e.g. angry,happy,neutral,sad) ----
    if getattr(args, 'emotions', None):
        emotions_filter = set(e.strip() for e in args.emotions.split(',') if e.strip())
        common_e2v = common_e2v & emotions_filter
        test_emotions_to_use = [e for e in test_emotions_to_use if eval_emotion_to_e2v.get(e) in common_e2v]
        eval_emotion_to_e2v = {e: eval_emotion_to_e2v[e] for e in test_emotions_to_use}
        logger.info(f"Restricted to emotions: {sorted(emotions_filter)} -> common_e2v={sorted(common_e2v)}, test_emotions={test_emotions_to_use}")

    logger.info(f"Common E2V emotions ({len(common_e2v)}): {sorted(common_e2v)}")
    if not common_e2v:
        logger.error("No common emotions!")
        return

    # ---- Build valid label list ----
    # For e2v: use E2V_LABELS ordering, restrict to common emotions
    # For wavlm/hubert: use the test dataset's linear probe ordering
    if ssl_model == 'e2v':
        # E2V head has 9 classes -- restrict to common
        seen = set()
        valid_e2v_labels = []
        for e in test_emotions_to_use:
            e2v = eval_emotion_to_e2v.get(e)
            if e2v and e2v not in seen and e2v in E2V_LABELS:
                valid_e2v_labels.append(e2v)
                seen.add(e2v)
        valid_indices = [E2V_LABELS.index(l) for l in valid_e2v_labels]
        valid_labels = valid_e2v_labels
    else:
        # Linear probe -- load test dataset's classifier
        cls_path = Path(args.classifier_dir) / ssl_model / args.test_dataset / 'best_model.pt'
        if not cls_path.exists():
            logger.error(f"Classifier not found: {cls_path}")
            return
        probe_ckpt = torch.load(cls_path, map_location='cpu', weights_only=False)
        probe_emotions = probe_ckpt['emotions']
        probe_e2v_map = test_config.emotion_to_e2v
        probe_e2v_labels = [probe_e2v_map.get(e) for e in probe_emotions]
        valid_indices = []
        valid_labels = []
        seen = set()
        for i, (emo, e2v_label) in enumerate(zip(probe_emotions, probe_e2v_labels)):
            if e2v_label in common_e2v and e2v_label not in seen:
                valid_indices.append(i)
                valid_labels.append(e2v_label)
                seen.add(e2v_label)

    logger.info(f"Valid classes ({len(valid_labels)}): {valid_labels}")
    logger.info(f"Valid indices: {valid_indices}")

    # ---- Load classifier ----
    if ssl_model == 'e2v':
        classifier = load_e2v_head(E2V_HEAD_PATH, device)
        classify_fn = classify_e2v
    else:
        cls_path = Path(args.classifier_dir) / ssl_model / args.test_dataset / 'best_model.pt'
        classifier = load_linear_probe(str(cls_path), device)
        classify_fn = classify_linear_probe

    # ---- Load SSL extractor ----
    extractor = get_ssl_extractor(ssl_model, device=device)

    # ---- Load test data ----
    splits_dir = Path(args.splits_dir) / args.test_dataset
    with open(splits_dir / 'test.json') as f:
        test_files = json.load(f)

    filtered = {e: test_files.get(e, []) for e in test_emotions_to_use if e in test_files}

    # ---- Pre-extract features ----
    logger.info("\nPre-extracting features...")
    cached_by_emotion = pre_extract_by_emotion(
        extractor, filtered, eval_emotion_to_e2v, common_e2v,
        max_samples=args.max_samples,
    )
    del extractor
    torch.cuda.empty_cache()

    total_samples = sum(len(v) for v in cached_by_emotion.values())
    logger.info(f"Total: {total_samples} samples across {len(cached_by_emotion)} emotions")
    if total_samples == 0:
        logger.error("No samples extracted!")
        return

    # ---- Evaluate (per-sample) ----
    logger.info(f"\n{'='*40}")
    logger.info(f"Evaluating {ssl_model} balanced codebook (per-sample)")
    logger.info(f"{'='*40}")

    samples = evaluate_per_sample(
        model=model,
        cached_by_emotion=cached_by_emotion,
        classifier=classifier,
        classify_fn=classify_fn,
        valid_indices=valid_indices,
        valid_labels=valid_labels,
        num_layers=args.num_layers,
        device=device,
        metric=args.metric,
    )

    # ---- Save results ----
    metric_suffix = f'_{args.metric}' if args.metric != 'cosine' else ''
    output_dir = Path(args.output_dir + metric_suffix) / ssl_model
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_id:
        output_file = output_dir / f'{args.codebook_dataset}_id.json'
    else:
        output_file = output_dir / f'{args.codebook_dataset}_to_{args.test_dataset}_ood.json'

    output_data = {
        'config': {
            'ssl_model': ssl_model,
            'codebook_dataset': args.codebook_dataset,
            'test_dataset': args.test_dataset,
            'eval_type': eval_type,
            'num_layers': args.num_layers,
            'max_samples': args.max_samples,
            'common_e2v_emotions': sorted(common_e2v),
            'test_emotions_used': test_emotions_to_use,
            'valid_labels': valid_labels,
        },
        'samples': samples,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\n Results saved to {output_file}")

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print(f"Summary: {ssl_model} {args.codebook_dataset} -> {args.test_dataset} ({eval_type})")
    print(f"  Total samples: {len(samples)}")
    print(f"{'='*80}")

    # Compute per-emotion aggregate for quick check
    for emo in valid_labels:
        emo_samples = [s for s in samples if s['true_label'] == emo]
        if not emo_samples:
            continue
        n = len(emo_samples)
        # Check a few layers
        for layer_idx in [0, 3, 7, 15, 31]:
            if layer_idx >= args.num_layers:
                continue
            avg_cos = np.mean([s['cosines'][layer_idx] for s in emo_samples])
            recall = np.mean([1.0 if s['preds'][layer_idx] == emo else 0.0 for s in emo_samples])
            print(f"  {emo:>12s} (n={n}) L{layer_idx+1:2d}: cos={avg_cos:.4f}, recall={recall:.4f}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
