#!/usr/bin/env python3
"""
RQ4 Ratio Post-Processor -- Compute Macro-F1 from existing sample_data.json

Reads per-sample records and tuned params produced by rq4_ratio_evaluate.py,
re-applies all 9 method decisions, and computes sklearn macro-F1 (4 fair
emotions) for each (config, source, codebook_type, OOD_dataset) combination.

Replaces the accuracy-based methods.json with F1-based methods_f1.json.

Usage:
  python -m paper_pipeline.evaluators.rq4_ratio_compute_f1 \
      --codebook-config 2x32 --codebook-source esd_en

  # Run all configs × sources:
  python -m paper_pipeline.evaluators.rq4_ratio_compute_f1 --all
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from core.config import FAIR_EMOTIONS, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']
OOD_TARGETS = ['msp', 'cameo_emns', 'cameo_enterface', 'cameo_jl_corpus']

CODEBOOK_TYPES = ['biased', 'mixed_r99']

METHOD_ORDER = [
    'Baseline', 'Bal_LS', 'Bal_Select',
    'Unfilt', 'Unfilt_ID',
    'Filt', 'Filt_ID', 'Filt_FB_LS', 'Filt_FBsel',
]

RQ4_RATIO_DIR = RESULTS_DIR / 'rq4_ratio'


def _predict_methods(
    records: List[dict],
    params: dict,
    cb_type_key: str,
) -> tuple:
    """Re-apply all 9 methods and return (trues, {method: preds})."""
    trues = []
    preds = {m: [] for m in METHOD_ORDER}

    sl_ls = str(params['Bal_LS']['params']['layer'])
    sl_sel = str(params['Bal_Select']['params']['layer'])
    sl_unfilt = str(params['Unfilt']['params']['layer'])
    sl_unfilt_id = str(params['Unfilt_ID']['params']['layer'])
    sl_filt = str(params['Biased_Filt']['params']['layer'])
    sl_filt_id = str(params['Filt_ID']['params']['layer'])
    sl_fbsel = str(params.get('Bal_Select_CV', params['Bal_Select'])['params']['layer'])

    for r in records:
        true_label = r['true_label']
        trues.append(true_label)

        bl_pred = r['baseline']['prediction']
        preds['Baseline'].append(bl_pred)

        bal = r.get('balanced', {})
        preds['Bal_LS'].append(bal.get(sl_ls, {}).get('prediction', bl_pred))
        preds['Bal_Select'].append(bal.get(sl_sel, {}).get('prediction', bl_pred))

        def _argmax_cosine(sl):
            best_emo, best_cos = bl_pred, -float('inf')
            emo_data = r.get(cb_type_key, {})
            for emo, ed in emo_data.items():
                if sl in ed and ed[sl]['cosine'] > best_cos:
                    best_cos = ed[sl]['cosine']
                    best_emo = emo
            return best_emo

        preds['Unfilt'].append(_argmax_cosine(sl_unfilt))
        preds['Unfilt_ID'].append(_argmax_cosine(sl_unfilt_id))

        def _filt(sl, fallback_pred):
            bal_cos = bal.get(sl, {}).get('cosine', None)
            if bal_cos is None:
                return fallback_pred
            emo_data = r.get(cb_type_key, {})
            best_emo, best_delta = None, -float('inf')
            for emo, ed in emo_data.items():
                if sl in ed:
                    delta = ed[sl]['cosine'] - bal_cos
                    if delta > best_delta:
                        best_delta = delta
                        best_emo = emo
            if best_delta <= 0 or best_emo is None:
                return fallback_pred
            return best_emo

        bal_ls_pred = bal.get(sl_ls, {}).get('prediction', bl_pred)
        bal_sel_pred = bal.get(sl_fbsel, {}).get('prediction', bl_pred)

        preds['Filt'].append(_filt(sl_filt, bl_pred))
        preds['Filt_ID'].append(_filt(sl_filt_id, bl_pred))
        preds['Filt_FB_LS'].append(_filt(sl_filt, bal_ls_pred))
        preds['Filt_FBsel'].append(_filt(sl_filt, bal_sel_pred))

    return trues, preds


def _macro_f1(trues, preds) -> float:
    if not trues:
        return 0.0
    return float(f1_score(
        trues, preds, labels=FAIR_EMOTIONS, average='macro', zero_division=0))


def process_one(cb_config: str, cb_source: str):
    """Compute F1 for one (config, source) pair from existing sample_data."""
    base_dir = RQ4_RATIO_DIR / cb_config / cb_source

    sample_path = base_dir / 'sample_data.json'
    params_path = base_dir / 'params.json'

    if not sample_path.exists():
        logger.warning(f"No sample_data: {sample_path}")
        return
    if not params_path.exists():
        logger.warning(f"No params: {params_path}")
        return

    logger.info(f"Processing {cb_config}/{cb_source}")

    with open(sample_path) as f:
        all_data = json.load(f)
    with open(params_path) as f:
        all_params = json.load(f)

    method_results = {}

    for cb_type in CODEBOOK_TYPES:
        if cb_type not in all_params:
            logger.warning(f"  No params for {cb_type}, skipping")
            continue

        ood_params = all_params[cb_type]['ood']
        id_params = all_params[cb_type]['id']

        type_results = {'id': {}, 'ood': {}}

        for ds, records in all_data.items():
            if not records:
                continue

            is_id = (ds == cb_source)
            params = id_params if is_id else ood_params

            trues, preds = _predict_methods(records, params, cb_type)

            ds_f1 = {}
            for method in METHOD_ORDER:
                ds_f1[method] = _macro_f1(trues, preds[method])

            if is_id:
                type_results['id'][ds] = ds_f1
            else:
                type_results['ood'][ds] = ds_f1

        ood_group = type_results['ood']
        if ood_group:
            avg = {}
            for m in METHOD_ORDER:
                vals = [ood_group[ds][m] for ds in ood_group if m in ood_group[ds]]
                if vals:
                    avg[m] = float(np.mean(vals))
            type_results['ood_avg'] = avg

        method_results[cb_type] = type_results

        ood_avg = type_results.get('ood_avg', {})
        logger.info(f"  {cb_type}:")
        for m in METHOD_ORDER:
            v = ood_avg.get(m, 0) * 100
            logger.info(f"    {m:>15s}: F1={v:6.2f}%")

    out_path = base_dir / 'methods_f1.json'
    with open(out_path, 'w') as f:
        json.dump(method_results, f, indent=2)
    logger.info(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='RQ4 Ratio: Compute Macro-F1 from sample_data.json')
    parser.add_argument('--codebook-config', type=str)
    parser.add_argument('--codebook-source', type=str, choices=ID_DATASETS)
    parser.add_argument('--all', action='store_true',
                        help='Process all available (config, source) pairs')
    args = parser.parse_args()

    if args.all:
        for config_dir in sorted(RQ4_RATIO_DIR.iterdir()):
            if not config_dir.is_dir():
                continue
            for source_dir in sorted(config_dir.iterdir()):
                if not source_dir.is_dir():
                    continue
                if (source_dir / 'sample_data.json').exists():
                    process_one(config_dir.name, source_dir.name)
    elif args.codebook_config and args.codebook_source:
        process_one(args.codebook_config, args.codebook_source)
    else:
        parser.error("Provide --codebook-config and --codebook-source, or --all")


if __name__ == '__main__':
    main()
