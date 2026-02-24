#!/usr/bin/env python3
"""
RQ2.1 Figure -- SER F1-Macro: Matched vs Unmatched vs Balanced (e2v, OOD)

Single plot:
  - X-axis: RVQ Layer (1-32)
  - Y-axis: SER F1-Macro
  - 3 lines: Biased (matched), Biased (unmatched), Balanced
  - Data: averaged over all OOD (cross-dataset) pairs

Reads from:  results/rq2_1_matched_ser/{src}_to_{tgt}_ood.json
Outputs to:  results/paper_figures/rq2_1_ser_f1_ood.png  (+.pdf)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
EXP_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from config import RESULTS_DIR, PAPER_FIGURES_DIR, ID_DATASETS, NUM_LAYERS

RESULTS_INPUT = RESULTS_DIR / 'rq2_1_matched_ser_128x8'

CODEBOOK_TYPES = {
    'biased_matched': {
        'label': 'Biased (matched)',
        'color': '#1E88E5',
        'marker': 'o',
        'ls': '-',
    },
    'biased_unmatched': {
        'label': 'Biased (unmatched)',
        'color': '#E53935',
        'marker': 'o',
        'ls': '-',
    },
    'balanced': {
        'label': 'Balanced',
        'color': '#9E9E9E',
        'marker': 'o',
        'ls': '-',
    },
}


def _load_ood_and_avg() -> dict:
    """Load all OOD result JSONs and average F1-macro per layer per codebook type."""
    all_data = []
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src == tgt:
                continue
            p = RESULTS_INPUT / f'{src}_to_{tgt}_ood.json'
            if p.exists():
                with open(p) as f:
                    all_data.append(json.load(f))

    if not all_data:
        return {}

    result = {}
    for ct in CODEBOOK_TYPES:
        layer_vals = defaultdict(list)
        for d in all_data:
            ct_data = d.get(ct, {})
            for layer_key, metrics in ct_data.items():
                layer_num = int(layer_key.replace('layer_', ''))
                layer_vals[layer_num].append(metrics.get('f1_macro', 0.0))
        result[ct] = {l: float(np.mean(vs)) for l, vs in sorted(layer_vals.items())}

    return result


def _plot_figure(output_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    data = _load_ood_and_avg()

    if not data:
        print("  [WARNING] No OOD data found in", RESULTS_INPUT)
        print("  Run the evaluator first:  python -m paper_pipeline.pipeline --rq 2.1 --eval")
        return

    layers = list(range(1, NUM_LAYERS + 1))

    for ct_key, style in CODEBOOK_TYPES.items():
        vals = data.get(ct_key, {})
        xs = sorted(vals.keys())
        ys = [vals[x] for x in xs]
        if ys:
            ax.plot(xs, ys,
                    label=style['label'],
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['ls'],
                    linewidth=2.5,
                    markersize=8)

    ax.set_xlabel('RVQ Layer', fontsize=16)
    ax.set_ylabel('SER F1-Macro', fontsize=16)
    ax.set_xticks(list(range(1, NUM_LAYERS + 1)))
    ax.set_xlim(0.5, NUM_LAYERS + 0.5)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13, loc='lower right')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path.with_suffix('.pdf')), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = PAPER_FIGURES_DIR / 'rq2_1_ser_f1_ood.png'

    if dry_run:
        print(f"  [DRY RUN] Would generate {output}")
        return

    _plot_figure(output)


def description() -> str:
    return "RQ2.1: SER F1-Macro — Matched vs Unmatched vs Balanced (e2v, OOD)"
