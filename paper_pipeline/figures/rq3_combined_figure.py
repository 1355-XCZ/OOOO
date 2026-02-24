#!/usr/bin/env python3
"""
RQ3 Combined Figure -- Layerwise F1-Macro for Ratio + Ambiguity codebooks

Full version (1 row × 6):
  (a) Ratio: [e2v] [HuBERT] [WavLM]  -- 5 lines each
  (b) Ambiguity: [e2v] [HuBERT] [WavLM]  -- 3 lines each

e2v-only variant (1 row × 2):
  (a) Ratio: [e2v]  -- 5 lines
  (b) Ambiguity: [e2v]  -- 3 lines

Usage:
  python -m paper_pipeline.figures.rq3_combined_figure              # full
  python -m paper_pipeline.figures.rq3_combined_figure --e2v-only   # variant
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
EXP_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from config import RESULTS_DIR, PAPER_FIGURES_DIR, ID_DATASETS

NUM_LAYERS = 24
CODEBOOK_DATASET_AMB = 'iemocap'
OOD_TARGETS_AMB = ['esd_en', 'ravdess', 'cremad']

SSL_CONFIGS = [
    {'ssl': 'e2v',    'cb_config': '2x24',    'display': 'emotion2vec'},
    {'ssl': 'hubert', 'cb_config': '1024x24', 'display': 'HuBERT'},
    {'ssl': 'wavlm',  'cb_config': '1024x24', 'display': 'WavLM'},
]

RATIO_KEYS = [
    'balanced', 'mixed_r10', 'mixed_r20', 'mixed_r40', 'mixed_r50',
    'mixed_r70', 'mixed_r80', 'mixed_r95', 'mixed_r99', 'biased',
]
RATIO_STYLES = {
    'balanced':  {'label': 'Balanced',  'color': '#9E9E9E', 'marker': 'o', 'ls': '-'},
    'mixed_r10': {'label': '10+90',     'color': '#AB47BC', 'marker': 'p', 'ls': '-'},
    'mixed_r20': {'label': '20+80',     'color': '#7E57C2', 'marker': 'h', 'ls': '-'},
    'mixed_r40': {'label': '40+60',     'color': '#5C6BC0', 'marker': '*', 'ls': '-'},
    'mixed_r50': {'label': '50+50',     'color': '#29B6F6', 'marker': 'X', 'ls': '-'},
    'mixed_r70': {'label': '70+30',     'color': '#26A69A', 'marker': 'P', 'ls': '-'},
    'mixed_r80': {'label': '80+20',     'color': '#42A5F5', 'marker': 's', 'ls': '-'},
    'mixed_r95': {'label': '95+5',      'color': '#66BB6A', 'marker': '^', 'ls': '-'},
    'mixed_r99': {'label': '99+1',      'color': '#FFA726', 'marker': 'D', 'ls': '-'},
    'biased':    {'label': 'Biased',    'color': '#EF5350', 'marker': 'v', 'ls': '-'},
}

AMBIGUITY_KEYS = ['high', 'mid', 'low']
AMBIGUITY_STYLES = {
    'high': {'label': 'Low Amb. (0%)',   'color': '#66BB6A', 'marker': 'o', 'ls': '-'},
    'mid':  {'label': 'Mid Amb. (20%)',  'color': '#FFA726', 'marker': 's', 'ls': '-'},
    'low':  {'label': 'High Amb. (33%)', 'color': '#EF5350', 'marker': '^', 'ls': '-'},
}


# ============================================================
# Data loading
# ============================================================

def _load_ratio_layerwise(ssl: str, cb_config: str) -> dict:
    """Load RQ3.1 ratio OOD results, return per-ratio per-layer (mean, std) of F1-Macro."""
    ssl_dir = RESULTS_DIR / 'rq3_1_ratio' / cb_config / ssl
    all_data = []
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src == tgt:
                continue
            p = ssl_dir / f'{src}_to_{tgt}_ood.json'
            if p.exists():
                with open(p) as f:
                    all_data.append(json.load(f))

    if not all_data:
        return {}

    result = {}
    for rk in RATIO_KEYS:
        means, stds = [], []
        for l in range(1, NUM_LAYERS + 1):
            lk = f'layer_{l}'
            vals = [d.get(rk, {}).get(lk, {}).get('f1_macro', 0.0)
                    for d in all_data if d.get(rk, {}).get(lk)]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if vals else 0.0)
        result[rk] = {'mean': means, 'std': stds}
    return result


def _load_ambiguity_layerwise(ssl: str, cb_config: str) -> dict:
    """Load RQ3.2 ambiguity OOD results, return per-level per-layer (mean, std) of F1-Macro."""
    ssl_dir = RESULTS_DIR / 'rq3_2_ambiguity' / cb_config / ssl
    all_data = []
    for tgt in OOD_TARGETS_AMB:
        p = ssl_dir / f'{CODEBOOK_DATASET_AMB}_to_{tgt}_ood.json'
        if p.exists():
            with open(p) as f:
                all_data.append(json.load(f))

    if not all_data:
        return {}

    result = {}
    for ak in AMBIGUITY_KEYS:
        means, stds = [], []
        for l in range(1, NUM_LAYERS + 1):
            lk = f'layer_{l}'
            vals = [d.get(ak, {}).get(lk, {}).get('f1_macro', 0.0)
                    for d in all_data if d.get(ak, {}).get(lk)]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if vals else 0.0)
        result[ak] = {'mean': means, 'std': stds}
    return result


# ============================================================
# Plotting
# ============================================================

def _plot(ssl_list, output_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_ssl = len(ssl_list)
    n_cols = n_ssl * 2
    fig = plt.figure(figsize=(7.0 * n_cols, 5.5))

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[n_ssl, n_ssl], wspace=0.18)
    gs_ratio = gridspec.GridSpecFromSubplotSpec(1, n_ssl, subplot_spec=gs[0], wspace=0.30)
    gs_amb = gridspec.GridSpecFromSubplotSpec(1, n_ssl, subplot_spec=gs[1], wspace=0.30)

    layers = list(range(1, NUM_LAYERS + 1))

    ratio_axes = []
    amb_axes = []

    # ---- (a) Ratio ----
    for col_idx, cfg in enumerate(ssl_list):
        ax = fig.add_subplot(gs_ratio[col_idx])
        ratio_axes.append(ax)
        data = _load_ratio_layerwise(cfg['ssl'], cfg['cb_config'])

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
        else:
            for rk in RATIO_KEYS:
                st = RATIO_STYLES[rk]
                ys = [data[rk].get(l, 0) for l in layers]
                ax.plot(layers, ys,
                        label=st['label'], color=st['color'],
                        marker=st['marker'], linestyle=st['ls'],
                        linewidth=2.5, markersize=6)

        ax.set_title(cfg['display'], fontsize=15, fontweight='bold')
        ax.set_xlabel('RVQ Layer', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel('SER F1-Macro', fontsize=13)
        ax.set_xticks(layers[::2])
        ax.set_xlim(0.5, NUM_LAYERS + 0.5)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower right')

    # ---- (b) Ambiguity ----
    for col_idx, cfg in enumerate(ssl_list):
        ax = fig.add_subplot(gs_amb[col_idx])
        amb_axes.append(ax)
        data = _load_ambiguity_layerwise(cfg['ssl'], cfg['cb_config'])

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
        else:
            for ak in AMBIGUITY_KEYS:
                st = AMBIGUITY_STYLES[ak]
                ys = [data[ak].get(l, 0) for l in layers]
                ax.plot(layers, ys,
                        label=st['label'], color=st['color'],
                        marker=st['marker'], linestyle=st['ls'],
                        linewidth=2.5, markersize=6)

        ax.set_title(cfg['display'], fontsize=15, fontweight='bold')
        ax.set_xlabel('RVQ Layer', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel('SER F1-Macro', fontsize=13)
        ax.set_xticks(layers[::2])
        ax.set_xlim(0.5, NUM_LAYERS + 0.5)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower right')

    # Section headers
    fig.canvas.draw()

    r_left = ratio_axes[0].get_position().x0
    r_right = ratio_axes[-1].get_position().x1
    fig.text((r_left + r_right) / 2, 1.03, '(a) Emotion Ratio',
             ha='center', fontsize=16, fontweight='bold', transform=fig.transFigure)

    a_left = amb_axes[0].get_position().x0
    a_right = amb_axes[-1].get_position().x1
    fig.text((a_left + a_right) / 2, 1.03, '(b) Annotator Ambiguity',
             ha='center', fontsize=16, fontweight='bold', transform=fig.transFigure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path.with_suffix('.pdf')), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    full_out = PAPER_FIGURES_DIR / 'rq3_combined.png'
    e2v_out = PAPER_FIGURES_DIR / 'rq3_combined_e2v.png'

    if dry_run:
        print(f"  [DRY RUN] Would generate {full_out}")
        print(f"  [DRY RUN] Would generate {e2v_out}")
        return

    _plot(SSL_CONFIGS, full_out)
    _plot([SSL_CONFIGS[0]], e2v_out)


def description():
    return "RQ3 Combined: Layerwise F1-Macro for Ratio + Ambiguity (full + e2v-only)"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description())
    parser.add_argument('--e2v-only', action='store_true',
                        help='Generate e2v-only variant only')
    args = parser.parse_args()

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.e2v_only:
        _plot([SSL_CONFIGS[0]], PAPER_FIGURES_DIR / 'rq3_combined_e2v.png')
    else:
        _plot(SSL_CONFIGS, PAPER_FIGURES_DIR / 'rq3_combined.png')
        _plot([SSL_CONFIGS[0]], PAPER_FIGURES_DIR / 'rq3_combined_e2v.png')
