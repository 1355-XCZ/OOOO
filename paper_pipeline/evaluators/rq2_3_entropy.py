#!/usr/bin/env python3
"""
RQ2.3 Evaluator -- Codebook Token Entropy & Utilization

For each codebook (balanced + 4 biased), compute normalized entropy and
utilization of token usage at each RVQ layer, broken down by test emotion.

Normalized entropy:  H~ = H / log(K)  in [0, 1]
Utilization:         U  = |unique tokens used| / K

Supports multiple SSL models and codebook configs.

Output per (ssl_model, codebook_dataset, test_dataset) pair:
  results/rq2_entropy_{config}/{ssl_model}/{src}_to_{tgt}_ood.json

Can be run standalone:
  python -m paper_pipeline.evaluators.rq2_3_entropy \
      --ssl-model e2v --codebook-config 2x24 \
      --codebook-dataset esd_en --test-dataset iemocap

Or via pipeline:
  python -m paper_pipeline.pipeline --rq 2.3 --eval
"""

import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.dataset_config import DATASET_CONFIGS
from core.config import CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR
from core.features import get_ssl_extractor, extract_features, get_codebook_dir
from core.quantize import load_codebook, EncoderDecoderRVQ

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']

DEFAULT_SSL_MODEL = 'e2v'
DEFAULT_CB_CONFIG = '2x24'
DEFAULT_NUM_LAYERS = 24
MAX_SAMPLES_DEFAULT = 200


def _result_subdir(config: str, ssl_model: str) -> str:
    return f'rq2_entropy_{config}/{ssl_model}'


def compute_normalized_entropy(counts: List[int]) -> float:
    """H / log(K) where K = len(counts). Returns value in [0, 1]."""
    total = sum(counts)
    if total == 0:
        return 0.0
    k = len(counts)
    if k <= 1:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(k)


def pre_extract_features(extractor, file_list: List[str], max_samples: int = 200):
    """Extract SSL features for up to max_samples files."""
    n_samples = min(len(file_list), max_samples)
    sampled = file_list[:n_samples]
    cached = []
    for audio_path in tqdm(sampled, desc="Pre-extracting features", leave=False):
        feats = extract_features(extractor, audio_path)
        if feats is not None:
            cached.append(feats)
    logger.info(f"  Cached {len(cached)}/{n_samples} samples")
    return cached


def collect_token_counts(
    codebook, cached_features: List[torch.Tensor],
    num_layers: int, codebook_size: int,
    device: str = 'cuda', desc: str = "Counting tokens",
) -> Dict[int, List[int]]:
    """Per-layer token counts across all samples."""
    counts = {l: [0] * codebook_size for l in range(1, num_layers + 1)}
    is_enc_dec = isinstance(codebook, EncoderDecoderRVQ)

    for features in tqdm(cached_features, desc=desc, leave=False):
        orig = features.to(device)
        feat_input = orig.unsqueeze(0) if orig.dim() == 2 else orig

        with torch.no_grad():
            if is_enc_dec:
                encoded = codebook.encoder(feat_input)
                _, indices, _ = codebook.rvq(encoded)
            else:
                _, indices, _, _ = codebook(feat_input)

            for layer in range(1, num_layers + 1):
                layer_indices = indices[:, :, layer - 1].flatten()
                for token_id in range(codebook_size):
                    counts[layer][token_id] += (layer_indices == token_id).sum().item()

    return counts


def evaluate_pair(
    codebook_dataset: str,
    test_dataset: str,
    ssl_model: str = DEFAULT_SSL_MODEL,
    cb_config: str = DEFAULT_CB_CONFIG,
    num_layers: int = DEFAULT_NUM_LAYERS,
    max_samples: int = MAX_SAMPLES_DEFAULT,
    device: str = 'cuda',
    output_dir: Optional[Path] = None,
) -> dict:
    """Evaluate entropy & utilization for one (source, target) pair."""

    is_id = (codebook_dataset == test_dataset)
    eval_type = "ID" if is_id else "OOD"

    cb_config_obj = DATASET_CONFIGS[codebook_dataset]
    test_config = DATASET_CONFIGS[test_dataset]

    cb_e2v_map = cb_config_obj.emotion_to_e2v
    test_e2v_map = test_config.emotion_to_e2v

    cb_e2v_labels = {cb_e2v_map[e] for e in cb_config_obj.emotions if e in cb_e2v_map}
    test_e2v_labels = {test_e2v_map[e] for e in test_config.emotions if e in test_e2v_map}
    common_e2v = cb_e2v_labels & test_e2v_labels

    if not common_e2v:
        logger.error(f"No overlapping emotions between {codebook_dataset} and {test_dataset}")
        return {}

    e2v_to_cb_emotion = {
        cb_e2v_map[e]: e for e in cb_config_obj.emotions
        if cb_e2v_map.get(e) in common_e2v
    }
    e2v_to_test_emotion = {
        test_e2v_map[e]: e for e in test_config.emotions
        if test_e2v_map.get(e) in common_e2v
    }
    eval_emotions = sorted(common_e2v)

    logger.info("=" * 60)
    logger.info(f"Codebook Token Entropy [{ssl_model}] ({eval_type})")
    logger.info(f"  Source: {codebook_dataset}  Target: {test_dataset}")
    logger.info(f"  Config: {cb_config}, Layers: {num_layers}")
    logger.info(f"  Common emotions: {eval_emotions}")
    logger.info("=" * 60)

    splits_dir = SPLITS_DIR / test_dataset
    with open(splits_dir / 'test.json') as f:
        test_splits = json.load(f)

    extractor = get_ssl_extractor(ssl_model, device=device)

    cb_dir = get_codebook_dir(str(CODEBOOK_DIR), ssl_model, codebook_dataset,
                              config=cb_config)
    logger.info(f"Codebook dir: {cb_dir}")

    balanced_codebook = load_codebook(str(cb_dir / 'balanced.pt'), device)

    biased_codebooks = {}
    for e2v_label in eval_emotions:
        cb_emotion = e2v_to_cb_emotion[e2v_label]
        cb = load_codebook(str(cb_dir / f'biased_{cb_emotion}.pt'), device)
        if cb:
            biased_codebooks[e2v_label] = cb

    if not balanced_codebook and not biased_codebooks:
        logger.error("No codebooks found!")
        return {}

    ref_model = balanced_codebook or next(iter(biased_codebooks.values()))
    actual_num_layers = ref_model.config.num_layers
    codebook_size = ref_model.config.codebook_size
    if actual_num_layers != num_layers:
        logger.warning(f"Codebook has {actual_num_layers} layers, using that instead of {num_layers}")
        num_layers = actual_num_layers
    logger.info(f"  Codebook params: L={num_layers}, K={codebook_size}")

    logger.info("Pre-extracting features...")
    cached_per_emotion = {}
    for e2v_label in eval_emotions:
        test_emotion = e2v_to_test_emotion[e2v_label]
        test_files = test_splits.get(test_emotion, [])
        if not test_files:
            logger.warning(f"No test files for {test_emotion} (E2V: {e2v_label})")
            continue
        logger.info(f"  {test_emotion} (E2V: {e2v_label})...")
        cached = pre_extract_features(extractor, test_files, max_samples)
        if cached:
            cached_per_emotion[e2v_label] = cached

    del extractor
    torch.cuda.empty_cache()
    logger.info(f"Cached {sum(len(v) for v in cached_per_emotion.values())} total samples.")

    codebook_names = []
    if balanced_codebook:
        codebook_names.append('balanced')
    for e2v_label in eval_emotions:
        if e2v_label in biased_codebooks:
            codebook_names.append(f'biased_{e2v_label}')

    entropy_results = {}
    utilization_results = {}

    for cb_name in codebook_names:
        if cb_name == 'balanced':
            cb_model = balanced_codebook
        else:
            e2v_label = cb_name.split('biased_', 1)[1]
            cb_model = biased_codebooks[e2v_label]

        entropy_results[cb_name] = {}
        utilization_results[cb_name] = {}
        logger.info(f"\n--- Codebook: {cb_name} ---")

        for emotion in eval_emotions:
            if emotion not in cached_per_emotion:
                continue

            cached = cached_per_emotion[emotion]
            counts = collect_token_counts(
                cb_model, cached, num_layers=num_layers,
                codebook_size=codebook_size, device=device,
                desc=f"{cb_name} / {emotion}",
            )

            layer_entropy = {}
            layer_utilization = {}
            for layer in range(1, num_layers + 1):
                h_tilde = compute_normalized_entropy(counts[layer])
                layer_entropy[str(layer)] = round(h_tilde, 6)
                unique_used = sum(1 for c in counts[layer] if c > 0)
                util = unique_used / codebook_size if codebook_size > 0 else 0.0
                layer_utilization[str(layer)] = round(util, 6)

            entropy_results[cb_name][emotion] = layer_entropy
            utilization_results[cb_name][emotion] = layer_utilization
            logger.info(f"  {emotion}: L1={layer_entropy['1']:.4f} ... "
                        f"L{num_layers}={layer_entropy[str(num_layers)]:.4f}")

    results = {
        'config': {
            'ssl_model': ssl_model,
            'codebook_config': cb_config,
            'codebook_dataset': codebook_dataset,
            'test_dataset': test_dataset,
            'eval_type': eval_type,
            'common_emotions': eval_emotions,
            'max_samples': max_samples,
            'num_layers': num_layers,
            'codebook_size': codebook_size,
        },
        'entropy': entropy_results,
        'utilization': utilization_results,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_id:
            out_path = output_dir / f'{codebook_dataset}_id.json'
        else:
            out_path = output_dir / f'{codebook_dataset}_to_{test_dataset}_ood.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {out_path}")

    return results


def run(dry_run: bool = False):
    """Evaluate all ID + OOD pairs for default config."""
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
            tag = f"{src}_id" if src == tgt else f"{src}_to_{tgt}_ood"
            print(f"    {tag}")
        return

    for src, tgt in pairs:
        tag = f"{src}_id" if src == tgt else f"{src}_to_{tgt}_ood"
        out_path = output_base / f"{tag}.json"
        if out_path.exists():
            logger.info(f"  [SKIP] {tag} (already exists)")
            continue
        evaluate_pair(src, tgt, output_dir=output_base)


def description() -> str:
    return "RQ2.3: Codebook Token Entropy & Utilization"


def main():
    parser = argparse.ArgumentParser(description=description())
    parser.add_argument('--ssl-model', type=str, default=DEFAULT_SSL_MODEL,
                        choices=['e2v', 'hubert', 'wavlm'])
    parser.add_argument('--codebook-config', type=str, default=DEFAULT_CB_CONFIG,
                        help='Codebook config (e.g. 2x24, 1024x24)')
    parser.add_argument('--codebook-dataset', type=str, required=True)
    parser.add_argument('--test-dataset', type=str, default=None)
    parser.add_argument('--num-layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES_DEFAULT)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.test_dataset is None:
        args.test_dataset = args.codebook_dataset

    if args.output_dir is None:
        args.output_dir = str(RESULTS_DIR / _result_subdir(args.codebook_config, args.ssl_model))

    evaluate_pair(
        codebook_dataset=args.codebook_dataset,
        test_dataset=args.test_dataset,
        ssl_model=args.ssl_model,
        cb_config=args.codebook_config,
        num_layers=args.num_layers,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
