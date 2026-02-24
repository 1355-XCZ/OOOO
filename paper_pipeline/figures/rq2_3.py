#!/usr/bin/env python3
"""
RQ2.3 Figure -- Codebook Token Entropy (OOD only, 128x8, e2v)

Single row of 5 subplots:
  Balanced | Biased (Angry) | Biased (Happy) | Biased (Neutral) | Biased (Sad)

Each subplot:
  - X-axis: RVQ Layer (1-8)
  - Y-axis: Normalized Entropy H/log(K)
  - 4 lines: one per test emotion (angry, happy, neutral, sad)
  - For biased codebooks: matched emotion = solid, unmatched = dashed

Data source: results/rq2_entropy_128x8/e2v/{src}_to_{tgt}_ood.json
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

ENTROPY_DIR = RESULTS_DIR / 'rq2_entropy_128x8' / 'e2v'

COMMON_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

EMOTION_COLORS = {
    'angry':   '#E53935',
    'happy':   '#FB8C00',
    'neutral': '#757575',
    'sad':     '#1E88E5',
}

EMOTION_DISPLAY = {
    'angry': 'Angry',
    'happy': 'Happy',
    'neutral': 'Neutral',
    'sad': 'Sad',
}

CODEBOOK_COLUMNS = ['balanced', 'biased_angry', 'biased_happy', 'biased_neutral', 'biased_sad']

COLUMN_TITLES = {
    'balanced':       'Balanced',
    'biased_angry':   'Biased (Angry)',
    'biased_happy':   'Biased (Happy)',
    'biased_neutral': 'Biased (Neutral)',
    'biased_sad':     'Biased (Sad)',
}


def _load_ood_entropy():
    """Load all OOD entropy JSONs and average per (codebook, emotion, layer)."""
    all_data = []
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src == tgt:
                continue
            p = ENTROPY_DIR / f'{src}_to_{tgt}_ood.json'
            if p.exists():
                with open(p) as f:
                    all_data.append(json.load(f))

    if not all_data:
        return {}

    result = {}
    for cb_name in CODEBOOK_COLUMNS:
        result[cb_name] = {}
        for emotion in COMMON_EMOTIONS:
            layer_vals = defaultdict(list)
            for d in all_data:
                ent = d.get('entropy', {}).get(cb_name, {}).get(emotion, {})
                for layer_str, val in ent.items():
                    layer_vals[int(layer_str)].append(val)
            result[cb_name][emotion] = {
                l: float(np.mean(vs)) for l, vs in sorted(layer_vals.items())
            }
    return result


def _plot_figure(output_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = _load_ood_entropy()
    if not data:
        print("  [WARNING] No OOD entropy data found in", ENTROPY_DIR)
        return

    n_cols = len(CODEBOOK_COLUMNS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5), sharey=True)

    for col_idx, cb_name in enumerate(CODEBOOK_COLUMNS):
        ax = axes[col_idx]
        cb_data = data.get(cb_name, {})
        is_biased = cb_name.startswith('biased_')
        biased_emotion = cb_name.split('biased_')[1] if is_biased else None

        for emotion in COMMON_EMOTIONS:
            ent = cb_data.get(emotion, {})
            xs = sorted(ent.keys())
            ys = [ent[x] for x in xs]
            if not ys:
                continue

            ls = '-'
            if is_biased:
                ls = '-' if emotion == biased_emotion else '--'

            ax.plot(xs, ys,
                    label=EMOTION_DISPLAY[emotion],
                    color=EMOTION_COLORS[emotion],
                    linestyle=ls,
                    linewidth=2.5,
                    marker='o' if (not is_biased or emotion == biased_emotion) else None,
                    markersize=6)

        ax.set_title(COLUMN_TITLES[cb_name], fontsize=16, fontweight='bold')
        ax.set_xlabel('RVQ Layer', fontsize=14)
        if col_idx == 0:
            ax.set_ylabel('Normalized Entropy', fontsize=14)
        ax.set_xticks(list(range(1, NUM_LAYERS + 1)))
        ax.set_xlim(0.5, NUM_LAYERS + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)

        if col_idx == n_cols - 1:
            ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path.with_suffix('.pdf')), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = PAPER_FIGURES_DIR / 'rq2_3_entropy_ood.png'

    if dry_run:
        print(f"  [DRY RUN] Would generate {output}")
        return

    _plot_figure(output)


def description() -> str:
    return "RQ2.3: Codebook Token Entropy — OOD (128x8, e2v)"
