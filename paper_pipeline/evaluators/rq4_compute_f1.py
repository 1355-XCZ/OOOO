#!/usr/bin/env python3
"""
RQ4 Post-Processor -- Compute Macro-F1 from existing sample_data

Reads raw per-sample records (produced by evaluate_unified.py) and
best_params_ood.json / best_params_id.json, re-applies all 9 method
decisions, and computes sklearn macro-F1 (4 fair emotions).

Usage:
  python -m paper_pipeline.evaluators.rq4_compute_f1 \
      --codebook-config 128x8 --codebook-source esd_en
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from core.config import FAIR_EMOTIONS, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']
OOD_DATASETS = ['msp', 'cameo_emns', 'cameo_enterface', 'cameo_jl_corpus']

METHOD_ORDER = [
    'Baseline', 'Bal_LS', 'Bal_Select', 'Unfilt', 'Unfilt_ID',
    'Filt', 'Filt_ID', 'Filt_FBLS', 'Filt_FBsel',
]


def _subdir_suffix(cb_config: str) -> str:
    return f'_{cb_config}' if cb_config != '2x32' else ''


def _sample_data_dir(cb_config: str, source: str) -> Path:
    return RESULTS_DIR / f'sample_data{_subdir_suffix(cb_config)}' / 'e2v' / source


def _unified_dir(cb_config: str, source: str) -> Path:
    return RESULTS_DIR / f'unified{_subdir_suffix(cb_config)}' / 'e2v' / source


def _last_layer(cb_config: str) -> int:
    return int(cb_config.split('x')[1])


def _load_params(cb_config: str, source: str, kind: str = 'ood') -> dict:
    udir = _unified_dir(cb_config, source)
    key = f'best_params_{kind}'

    # New format: unified_{config}/e2v/{source}.json (--phase all)
    new_path = udir.parent / f'{source}.json'
    if new_path.exists():
        with open(new_path) as f:
            blob = json.load(f)
        if key in blob:
            raw = blob[key]
            return {m: {'acc': d['val_acc'], 'params': d['params']}
                    for m, d in raw.items()}

    # Old format: unified_{config}/e2v/{source}/best_params_{kind}.json
    old_path = udir / f'{key}.json'
    if old_path.exists():
        with open(old_path) as f:
            raw = json.load(f)
        return {m: {'acc': d['val_acc'], 'params': d['params']}
                for m, d in raw.items()}

    logger.warning(f"Params not found: tried {new_path} and {old_path}")
    return {}


def _load_records(cb_config: str, source: str, dataset: str) -> list:
    p = _sample_data_dir(cb_config, source) / f'{dataset}.json'
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def _predict_methods(records: list, params: dict, last_layer: int) -> Dict[str, list]:
    """Re-apply all 9 methods and return per-sample predictions."""
    preds = {m: [] for m in METHOD_ORDER}
    trues = []

    sl_last = str(last_layer)
    sl_bal_sel = str(params.get('Bal_Select', {}).get('params', {}).get('layer', last_layer))
    sl_unfilt = str(params.get('Unfilt', {}).get('params', {}).get('layer', last_layer))
    sl_unfilt_id = str(params.get('Unfilt_ID', {}).get('params', {}).get('layer', last_layer))
    sl_filt = str(params.get('Biased_Filt', {}).get('params', {}).get('layer', last_layer))
    sl_filt_id = str(params.get('Filt_ID', {}).get('params', {}).get('layer', last_layer))
    sl_fbsel = str(params.get('Bal_Select_CV', params.get('Bal_Select', {})).get('params', {}).get('layer', last_layer))

    for r in records:
        true_label = r['true_label']
        trues.append(true_label)

        bl_pred = r['baseline']['prediction']
        preds['Baseline'].append(bl_pred)

        bal = r.get('balanced', {})
        preds['Bal_LS'].append(bal.get(sl_last, {}).get('prediction', bl_pred))
        preds['Bal_Select'].append(bal.get(sl_bal_sel, {}).get('prediction', bl_pred))

        # Unfilt: argmax cosine among biased codebooks
        def _argmax_biased(sl):
            best_emo, best_cos = bl_pred, -float('inf')
            for emo, ed in r['biased'].items():
                if sl in ed and ed[sl]['cosine'] > best_cos:
                    best_cos = ed[sl]['cosine']
                    best_emo = emo
            return best_emo

        preds['Unfilt'].append(_argmax_biased(sl_unfilt))
        preds['Unfilt_ID'].append(_argmax_biased(sl_unfilt_id))

        # Filt: delta-cosine with fallback
        def _filt(sl, fallback_pred):
            bal_cos = bal.get(sl, {}).get('cosine', None)
            if bal_cos is None:
                return fallback_pred
            best_emo, best_delta = None, -float('inf')
            for emo, ed in r['biased'].items():
                if sl in ed:
                    delta = ed[sl]['cosine'] - bal_cos
                    if delta > best_delta:
                        best_delta = delta
                        best_emo = emo
            if best_delta <= 0 or best_emo is None:
                return fallback_pred
            return best_emo

        bal_ls_pred = bal.get(sl_last, {}).get('prediction', bl_pred)
        bal_sel_pred = bal.get(sl_fbsel, {}).get('prediction', bl_pred)

        preds['Filt'].append(_filt(sl_filt, bl_pred))
        preds['Filt_ID'].append(_filt(sl_filt_id, bl_pred))
        preds['Filt_FBLS'].append(_filt(sl_filt, bal_ls_pred))
        preds['Filt_FBsel'].append(_filt(sl_filt, bal_sel_pred))

    return trues, preds


def _compute_f1(trues, preds) -> float:
    if not trues:
        return 0.0
    return float(f1_score(trues, preds, labels=FAIR_EMOTIONS, average='macro', zero_division=0))


def _compute_accuracy(trues, preds) -> float:
    if not trues:
        return 0.0
    return float(np.mean([t == p for t, p in zip(trues, preds)]))


def evaluate_config_source(cb_config: str, source: str) -> dict:
    """Evaluate all methods for one (config, source) pair across 4 OOD datasets.

    Returns: {method: {dataset: {'f1': float, 'acc': float}, 'ood_avg_f1': float}}
    """
    ll = _last_layer(cb_config)
    ood_params = _load_params(cb_config, source, 'ood')
    if not ood_params:
        logger.warning(f"No OOD params for {cb_config}/{source}, skipping")
        return {}

    results = {}
    for method in METHOD_ORDER:
        results[method] = {}

    for ds in OOD_DATASETS:
        records = _load_records(cb_config, source, ds)
        if not records:
            logger.warning(f"  No sample data: {cb_config}/{source}/{ds}")
            continue

        trues, preds = _predict_methods(records, ood_params, ll)

        for method in METHOD_ORDER:
            f1 = _compute_f1(trues, preds[method])
            acc = _compute_accuracy(trues, preds[method])
            results[method][ds] = {'f1': f1, 'accuracy': acc, 'n_samples': len(trues)}

    for method in METHOD_ORDER:
        f1_vals = [results[method][ds]['f1'] for ds in OOD_DATASETS if ds in results[method]]
        results[method]['ood_avg_f1'] = float(np.mean(f1_vals)) if f1_vals else 0.0
        results[method]['ood_avg_acc'] = float(np.mean(
            [results[method][ds]['accuracy'] for ds in OOD_DATASETS if ds in results[method]]
        )) if f1_vals else 0.0

    return results


def run_single(cb_config: str, source: str):
    """Run F1 computation for one (config, source) pair and save results."""
    logger.info(f"Computing F1: config={cb_config}, source={source}")
    results = evaluate_config_source(cb_config, source)
    if not results:
        return

    out_dir = RESULTS_DIR / 'rq4_f1' / cb_config
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{source}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved: {out_path}")

    for method in METHOD_ORDER:
        avg_f1 = results[method].get('ood_avg_f1', 0)
        logger.info(f"  {method:>15s}: F1={avg_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description='RQ4: Compute Macro-F1 from sample data')
    parser.add_argument('--codebook-config', type=str, required=True)
    parser.add_argument('--codebook-source', type=str, required=True, choices=ID_DATASETS)
    args = parser.parse_args()
    run_single(args.codebook_config, args.codebook_source)


if __name__ == '__main__':
    main()
