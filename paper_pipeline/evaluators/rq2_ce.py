#!/usr/bin/env python3
"""
RQ2 Supplementary -- Cross-Entropy Distribution Preservation

Measures how well codebook quantization preserves the full emotion
distribution by computing CE between annotator vote distributions
and the SER head's softmax output at each RVQ layer.

Setup:
  - Codebooks trained on 3 ID datasets (esd_en, ravdess, cremad)
  - Test on IEMOCAP (OOD) multi-annotator samples
  - Supports all 3 SSL models: e2v (2x24), hubert (1024x24), wavlm (1024x24)

For each codebook type (balanced + 4 biased) x each layer 1..N:
  - Group samples by primary emotion
  - Quantize -> SER head -> softmax -> p
  - Compute CE(y, p) with epsilon smoothing, where y = vote distribution
  - Average CE per emotion subset

Output:
  results/rq2_ce/{ssl_model}/{source}_ce_{version}.json

Usage:
  python -m paper_pipeline.evaluators.rq2_ce \
      --ssl-model e2v --codebook-source esd_en --version va
"""

import sys
import json
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from core.config import (
    E2V_HEAD_PATH, CODEBOOK_DIR, SPLITS_DIR, RESULTS_DIR,
    FAIR_EMOTIONS, FAIR_E2V_INDICES, DATASET_TO_FAIR_MAP,
    set_seed,
)
from core.features import get_ssl_extractor, extract_features
from core.quantize import load_codebook, EncoderDecoderRVQ
from core.classify import load_e2v_head, load_custom_head

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SSL_CONFIGS = {
    'e2v':    {'cb_config': '2x24',    'num_layers': 24},
    'hubert': {'cb_config': '1024x24', 'num_layers': 24},
    'wavlm':  {'cb_config': '1024x24', 'num_layers': 24},
}
CODEBOOK_SOURCES = ['esd_en', 'ravdess', 'cremad']
CLASSIFIER_SOURCE_MAP = {
    'cremad_clear': 'cremad',
    'cremad_ambig': 'cremad',
}
RATIO_LEVELS = [95, 99]
EPSILON = 1e-6


def votes_to_soft_label(votes: dict) -> np.ndarray:
    """Convert vote dict to 4-class probability vector [angry, happy, neutral, sad]."""
    vec = np.zeros(len(FAIR_EMOTIONS), dtype=np.float64)
    for emo, count in votes.items():
        if emo in FAIR_EMOTIONS:
            idx = FAIR_EMOTIONS.index(emo)
            vec[idx] += count
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def epsilon_smooth(dist: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """Apply epsilon smoothing and renormalize."""
    smoothed = np.maximum(dist, eps)
    return smoothed / smoothed.sum()


def cross_entropy(y: np.ndarray, p: np.ndarray, eps: float = EPSILON) -> float:
    """CE(y, p) = -sum(y_smooth * log(p_smooth)) with epsilon smoothing."""
    y_s = epsilon_smooth(y, eps)
    p_s = epsilon_smooth(p, eps)
    return -float(np.sum(y_s * np.log(p_s)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for already-smoothed distributions."""
    return float(np.sum(p * np.log(p / q)))


def js_divergence(y: np.ndarray, p: np.ndarray, eps: float = EPSILON) -> float:
    """JS(y, p) = 0.5*KL(y||m) + 0.5*KL(p||m), m=(y+p)/2. Bounded [0, log2]."""
    y_s = epsilon_smooth(y, eps)
    p_s = epsilon_smooth(p, eps)
    m = 0.5 * (y_s + p_s)
    return 0.5 * _kl(y_s, m) + 0.5 * _kl(p_s, m)


def top2_set_match(y: np.ndarray, p: np.ndarray) -> int:
    """1 if the top-2 emotion sets (by value) are identical, else 0."""
    gt_top2 = set(np.argsort(y)[-2:])
    pred_top2 = set(np.argsort(p)[-2:])
    return 1 if gt_top2 == pred_top2 else 0


def _get_probs_at_layer(model, layer, partial, indices, head, is_enc_dec):
    """Accumulate one RVQ layer, return 4-class softmax probs and updated partial."""
    layer_idx = indices[:, :, layer - 1]
    cb_weights = model.rvq.layers[layer - 1]._codebook.embed[0]
    partial = partial + cb_weights[layer_idx]

    if is_enc_dec:
        quantized = model.decoder(partial).squeeze(0)
    else:
        quantized = partial.squeeze(0)

    q_mean = quantized.mean(dim=0).unsqueeze(0)
    logits = head(q_mean)
    valid_logits = logits[:, FAIR_E2V_INDICES]
    probs = F.softmax(valid_logits, dim=-1).squeeze(0).cpu().numpy()
    return probs, partial


def evaluate_ce_on_samples(model, cached_samples, head, num_layers, device,
                           desc='', save_sample_probs=False):
    """Run one codebook on samples, returning per-layer CE values.

    Args:
        cached_samples: list of (features_tensor, soft_label_np)
        save_sample_probs: if True, also return per-sample probs at all layers

    Returns:
        layer_ces: {layer: [ce_value, ...]}
        sample_probs: [{layer: probs_np}, ...] per sample (only if save_sample_probs)
    """
    is_enc_dec = isinstance(model, EncoderDecoderRVQ)
    layer_ces: Dict[int, List[float]] = defaultdict(list)
    sample_probs = [] if save_sample_probs else None

    for features, y in tqdm(cached_samples, desc=desc, leave=False):
        orig = features.to(device)
        feat_input = orig.unsqueeze(0) if orig.dim() == 2 else orig
        per_layer_p = {} if save_sample_probs else None

        with torch.no_grad():
            if is_enc_dec:
                encoded = model.encoder(feat_input)
                _, indices, _ = model.rvq(encoded)
                partial = torch.zeros_like(encoded)
            else:
                _, indices, _, _ = model(feat_input)
                partial = torch.zeros_like(feat_input)

            for layer in range(1, num_layers + 1):
                probs, partial = _get_probs_at_layer(
                    model, layer, partial, indices, head, is_enc_dec)
                ce = cross_entropy(y, probs)
                layer_ces[layer].append(ce)
                if save_sample_probs:
                    per_layer_p[layer] = probs.copy()

        if save_sample_probs:
            sample_probs.append(per_layer_p)

    if save_sample_probs:
        return dict(layer_ces), sample_probs
    return dict(layer_ces)


def run(args=None, dry_run=False):
    if dry_run:
        print("  [DRY RUN] Would run RQ2/RQ3 CE evaluation for all SSL × source combinations")
        return

    if args is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--ssl-model', type=str, required=True,
                            choices=list(SSL_CONFIGS.keys()))
        parser.add_argument('--codebook-source', type=str, required=True,
                            choices=CODEBOOK_SOURCES)
        parser.add_argument('--version', type=str, default='va',
                            choices=['va', 'vb'])
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--use-custom-head', action='store_true')
        parser.add_argument('--save-samples', action='store_true')
        parser.add_argument('--include-ratio', action='store_true')
        args = parser.parse_args()

    set_seed()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    ssl_model = args.ssl_model
    ssl_cfg = SSL_CONFIGS[ssl_model]
    cb_config = ssl_cfg['cb_config']
    num_layers = ssl_cfg['num_layers']
    cb_source = args.codebook_source

    testset_path = SPLITS_DIR / 'iemocap' / f'secondary_emotion_{args.version}.json'
    if not testset_path.exists():
        logger.error(f'Test set not found: {testset_path}')
        logger.error('Run: python scripts/utils/prepare_secondary_emotion_testset.py')
        sys.exit(1)

    with open(testset_path) as f:
        testset = json.load(f)
    samples = testset['samples']
    logger.info(f'Loaded {len(samples)} samples (version={args.version})')

    cb_dir = CODEBOOK_DIR / ssl_model / cb_config / cb_source
    fair_map = DATASET_TO_FAIR_MAP.get(cb_source, {})
    inv_map = {}
    for orig, fair in fair_map.items():
        if fair in FAIR_EMOTIONS and fair not in inv_map:
            inv_map[fair] = orig

    balanced = load_codebook(str(cb_dir / 'balanced.pt'), device)
    biased = {}
    for fair_emo, orig_emo in inv_map.items():
        model = load_codebook(str(cb_dir / f'biased_{orig_emo}.pt'), device)
        if model:
            biased[fair_emo] = model

    ratio_models = {}
    if args.include_ratio:
        for r in RATIO_LEVELS:
            r_models = {}
            for fair_emo, orig_emo in inv_map.items():
                path = cb_dir / f'mixed_{orig_emo}_r{r}.pt'
                if path.exists():
                    m = load_codebook(str(path), device)
                    if m:
                        r_models[fair_emo] = m
            if r_models:
                ratio_models[r] = r_models
        logger.info(f'Ratio codebooks loaded: {sorted(ratio_models.keys())}')

    logger.info(f'Codebooks ({ssl_model}/{cb_config}/{cb_source}): '
                f'balanced={"OK" if balanced else "MISSING"}, '
                f'biased={list(biased.keys())}')

    classifier_source = CLASSIFIER_SOURCE_MAP.get(cb_source, cb_source)
    if ssl_model == 'e2v' and not args.use_custom_head:
        head = load_e2v_head(E2V_HEAD_PATH, device)
    else:
        head = load_custom_head(classifier_source, ssl_model, device)
        if ssl_model == 'e2v':
            logger.info('Using custom-trained e2v classifier instead of native head')

    extractor = get_ssl_extractor(ssl_model, device=device)

    logger.info('Pre-extracting features...')
    sample_features = []
    for s in tqdm(samples, desc='Extracting'):
        feats = extract_features(extractor, s['audio'])
        if feats is not None:
            y = votes_to_soft_label(s['votes'])
            sample_features.append({
                'features': feats,
                'primary': s['primary'],
                'y': y,
                'utt_id': s.get('utt_id', ''),
                'votes': s.get('votes', {}),
            })

    del extractor
    torch.cuda.empty_cache()
    logger.info(f'Extracted {len(sample_features)}/{len(samples)} samples')

    by_primary = defaultdict(list)
    for s in sample_features:
        by_primary[s['primary']].append(s)

    results = {}
    sample_records = [] if args.save_samples else None

    # --- Baseline: no quantization ---
    logger.info('Computing baseline (no quantization)...')
    baseline_ce = {}
    for emo in FAIR_EMOTIONS:
        ces = []
        for s in by_primary.get(emo, []):
            orig = s['features'].to(device)
            with torch.no_grad():
                feat_input = orig.unsqueeze(0) if orig.dim() == 2 else orig
                q_mean = feat_input.mean(
                    dim=-2 if feat_input.dim() == 3 else 0).unsqueeze(0)
                logits = head(q_mean)
                valid_logits = logits[:, FAIR_E2V_INDICES]
                probs = F.softmax(valid_logits, dim=-1).squeeze(0).cpu().numpy()
            ce_val = cross_entropy(s['y'], probs)
            js_val = js_divergence(s['y'], probs)
            t2m = top2_set_match(s['y'], probs)
            ces.append(ce_val)
            if sample_records is not None:
                sample_records.append({
                    'utt_id': s['utt_id'],
                    'primary': emo,
                    'votes': s['votes'],
                    'y': {e: round(float(s['y'][i]), 4) for i, e in enumerate(FAIR_EMOTIONS)},
                    'codebook': 'baseline',
                    'layer': 0,
                    'p': {e: round(float(probs[i]), 6) for i, e in enumerate(FAIR_EMOTIONS)},
                    'ce': round(ce_val, 4),
                    'js': round(js_val, 6),
                    'top2_match': t2m,
                })
        baseline_ce[emo] = float(np.mean(ces)) if ces else None
    results['baseline'] = baseline_ce

    # --- Balanced codebook ---
    if balanced:
        logger.info('Evaluating: balanced (all primary subsets)')
        bal_results = {}
        for emo in FAIR_EMOTIONS:
            emo_group = by_primary.get(emo, [])
            emo_samples = [(s['features'], s['y']) for s in emo_group]
            if not emo_samples:
                continue
            ret = evaluate_ce_on_samples(
                balanced, emo_samples, head, num_layers, device,
                desc=f'balanced/{emo}', save_sample_probs=(sample_records is not None))
            if sample_records is not None:
                layer_ces, sp = ret
                for idx, s in enumerate(emo_group):
                    for layer in sorted(sp[idx].keys()):
                        probs = sp[idx][layer]
                        sample_records.append({
                            'utt_id': s['utt_id'], 'primary': emo, 'votes': s['votes'],
                            'y': {e: round(float(s['y'][i]), 4) for i, e in enumerate(FAIR_EMOTIONS)},
                            'codebook': 'balanced', 'layer': layer,
                            'p': {e: round(float(probs[i]), 6) for i, e in enumerate(FAIR_EMOTIONS)},
                            'ce': round(cross_entropy(s['y'], probs), 4),
                            'js': round(js_divergence(s['y'], probs), 6),
                            'top2_match': top2_set_match(s['y'], probs),
                        })
            else:
                layer_ces = ret
            bal_results[emo] = {
                f'layer_{l}': float(np.mean(ces))
                for l, ces in sorted(layer_ces.items())
            }
        results['balanced'] = bal_results

    # --- Biased codebooks (matched primary) ---
    for emo in FAIR_EMOTIONS:
        if emo not in biased:
            logger.warning(f'Missing biased codebook for {emo}')
            continue
        emo_group = by_primary.get(emo, [])
        emo_samples = [(s['features'], s['y']) for s in emo_group]
        if not emo_samples:
            logger.warning(f'No samples for primary={emo}')
            continue

        logger.info(f'Evaluating: biased_{emo} ({len(emo_samples)} samples)')
        ret = evaluate_ce_on_samples(
            biased[emo], emo_samples, head, num_layers, device,
            desc=f'biased_{emo}', save_sample_probs=(sample_records is not None))
        if sample_records is not None:
            layer_ces, sp = ret
            for idx, s in enumerate(emo_group):
                for layer in sorted(sp[idx].keys()):
                    probs = sp[idx][layer]
                    sample_records.append({
                        'utt_id': s['utt_id'], 'primary': emo, 'votes': s['votes'],
                        'y': {e: round(float(s['y'][i]), 4) for i, e in enumerate(FAIR_EMOTIONS)},
                        'codebook': f'biased_{emo}', 'layer': layer,
                        'p': {e: round(float(probs[i]), 6) for i, e in enumerate(FAIR_EMOTIONS)},
                        'ce': round(cross_entropy(s['y'], probs), 4),
                        'js': round(js_divergence(s['y'], probs), 6),
                        'top2_match': top2_set_match(s['y'], probs),
                    })
        else:
            layer_ces = ret
        results[f'biased_{emo}'] = {
            emo: {
                f'layer_{l}': float(np.mean(ces))
                for l, ces in sorted(layer_ces.items())
            }
        }

    # --- Ratio (mixed) codebooks ---
    for r, r_models in sorted(ratio_models.items()):
        for emo in FAIR_EMOTIONS:
            if emo not in r_models:
                continue
            emo_group = by_primary.get(emo, [])
            emo_samples = [(s['features'], s['y']) for s in emo_group]
            if not emo_samples:
                continue

            cb_label = f'mixed_r{r}_{emo}'
            logger.info(f'Evaluating: {cb_label} ({len(emo_samples)} samples)')
            ret = evaluate_ce_on_samples(
                r_models[emo], emo_samples, head, num_layers, device,
                desc=cb_label, save_sample_probs=(sample_records is not None))
            if sample_records is not None:
                layer_ces, sp = ret
                for idx, s in enumerate(emo_group):
                    for layer in sorted(sp[idx].keys()):
                        probs = sp[idx][layer]
                        sample_records.append({
                            'utt_id': s['utt_id'], 'primary': emo, 'votes': s['votes'],
                            'y': {e: round(float(s['y'][i]), 4) for i, e in enumerate(FAIR_EMOTIONS)},
                            'codebook': cb_label, 'layer': layer,
                            'p': {e: round(float(probs[i]), 6) for i, e in enumerate(FAIR_EMOTIONS)},
                            'ce': round(cross_entropy(s['y'], probs), 4),
                            'js': round(js_divergence(s['y'], probs), 6),
                            'top2_match': top2_set_match(s['y'], probs),
                        })
            else:
                layer_ces = ret
            results[cb_label] = {
                emo: {
                    f'layer_{l}': float(np.mean(ces))
                    for l, ces in sorted(layer_ces.items())
                }
            }

    head_tag = 'custom' if args.use_custom_head else 'native'
    out_dir = RESULTS_DIR / 'rq2_ce' / f'{ssl_model}_{head_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{cb_source}_ce_{args.version}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'ssl_model': ssl_model,
            'head_type': head_tag,
            'codebook_source': cb_source,
            'config': cb_config,
            'version': args.version,
            'epsilon': EPSILON,
            'n_samples': len(sample_features),
            'per_primary_count': {e: len(by_primary[e]) for e in FAIR_EMOTIONS},
            'results': results,
        }, f, indent=2)
    logger.info(f'Saved: {out_path}')

    if sample_records is not None:
        sample_path = out_dir / f'{cb_source}_samples_{args.version}.json'
        with open(sample_path, 'w') as f:
            json.dump(sample_records, f, indent=1)
        logger.info(f'Saved {len(sample_records)} sample records: {sample_path}')

    print(f'\n{"="*60}')
    print(f'  CE Results: {ssl_model}/{cb_source} (version={args.version})')
    print(f'{"="*60}')
    for emo in FAIR_EMOTIONS:
        bl = baseline_ce.get(emo)
        bl_str = f'{bl:.4f}' if bl is not None else 'N/A'
        bal = results.get('balanced', {}).get(emo, {}).get(f'layer_{num_layers}')
        bal_str = f'{bal:.4f}' if bal is not None else 'N/A'
        bias = results.get(f'biased_{emo}', {}).get(emo, {}).get(f'layer_{num_layers}')
        bias_str = f'{bias:.4f}' if bias is not None else 'N/A'
        print(f'  {emo:>8s}  baseline={bl_str}  balanced_L{num_layers}={bal_str}  '
              f'biased_L{num_layers}={bias_str}')
    print(f'{"="*60}')


def main():
    parser = argparse.ArgumentParser(
        description='RQ2 Cross-Entropy Distribution Evaluator')
    parser.add_argument('--ssl-model', type=str, required=True,
                        choices=list(SSL_CONFIGS.keys()))
    parser.add_argument('--codebook-source', type=str, required=True,
                        choices=CODEBOOK_SOURCES)
    parser.add_argument('--version', type=str, default='va',
                        choices=['va', 'vb'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-custom-head', action='store_true',
                        help='Use custom-trained classifier instead of native e2v head')
    parser.add_argument('--save-samples', action='store_true',
                        help='Save per-sample predictions at all layers')
    parser.add_argument('--include-ratio', action='store_true',
                        help='Also evaluate ratio (mixed) codebooks')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
