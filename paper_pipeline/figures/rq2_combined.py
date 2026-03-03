#!/usr/bin/env python3
"""
RQ2 Combined Figure -- SER Recall + Entropy for e2v vs non-emotion target

Layout:
  Row 1 (SER Recall): 2 subplots
    Left:  emotion target (e2v, 2x24)
    Right: non-emotion target (HuBERT or WavLM, 1024x24)
    Lines: Balanced(all)[solid], Biased(all)[solid], Biased(matched)[dashed], Biased(unmatched)[dashed]

  Row 2 (Entropy): 4 subplots
    Col 1: Balanced entropy (e2v) -- 4 emotion lines
    Col 2: Merged biased entropy (e2v) -- Match vs Unmatch
    Col 3: Balanced entropy (target) -- 4 emotion lines
    Col 4: Merged biased entropy (target) -- Match vs Unmatch

Generated twice: once with HuBERT, once with WavLM as target.

Usage:
  python -m paper_pipeline.figures.rq2_combined --target-ssl hubert
  python -m paper_pipeline.figures.rq2_combined --target-ssl wavlm
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

E2V_CONFIG = '2x24'
TARGET_CONFIG = '1024x24'
NUM_LAYERS = 24

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

CODEBOOK_TYPE_STYLES = {
    'balanced': {
        'label': 'Balanced (all)',
        'color': '#757575',
        'marker': 'o',
        'ls': '-',
        'lw': 2.5,
    },
    'biased_matched': {
        'label': 'Emotion-specific (matched)',
        'color': '#1E88E5',
        'marker': '^',
        'ls': '--',
        'lw': 1.8,
    },
    'biased_unmatched': {
        'label': 'Emotion-specific (unmatched)',
        'color': '#E53935',
        'marker': 'v',
        'ls': '--',
        'lw': 1.8,
    },
}

SSL_DISPLAY = {
    'e2v': 'emotion2vec',
    'hubert': 'HuBERT',
    'wavlm': 'WavLM',
}

BASELINE_DIR = RESULTS_DIR / 'ssl_comparison_l2_64x8_emilia_ood'
FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']


def _load_recall_baseline(ssl_model: str) -> float:
    """Load unquantized macro-recall baseline, averaged across OOD test datasets."""
    ssl_dir = BASELINE_DIR / ssl_model
    macro_recalls = []
    for ds in ID_DATASETS:
        p = ssl_dir / f'unquantized_{ds}.json'
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        per_emo = data.get('per_emotion', {})
        recalls = [per_emo[e]['recall'] for e in FAIR_EMOTIONS if e in per_emo]
        if recalls:
            macro_recalls.append(float(np.mean(recalls)))
    return float(np.mean(macro_recalls)) if macro_recalls else None


# ============================================================
# Data Loading: SER-F1
# ============================================================

def _ser_results_dir(config: str, ssl_model: str) -> Path:
    return RESULTS_DIR / f'rq2_matched_ser_{config}' / ssl_model


def _load_ser_ood_avg(config: str, ssl_model: str) -> dict:
    """Load OOD SER recall-macro results, average across all OOD pairs."""
    results_dir = _ser_results_dir(config, ssl_model)
    all_data = []
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src == tgt:
                continue
            p = results_dir / f'{src}_to_{tgt}_ood.json'
            if p.exists():
                with open(p) as f:
                    all_data.append(json.load(f))

    if not all_data:
        return {}

    result = {}
    for ct in CODEBOOK_TYPE_STYLES:
        layer_vals = defaultdict(list)
        for d in all_data:
            ct_data = d.get(ct, {})
            for layer_key, metrics in ct_data.items():
                layer_num = int(layer_key.replace('layer_', ''))
                layer_vals[layer_num].append(metrics.get('recall_macro', 0.0))
        result[ct] = {l: float(np.mean(vs)) for l, vs in sorted(layer_vals.items())}

    return result


# ============================================================
# Data Loading: Entropy
# ============================================================

def _entropy_results_dir(config: str, ssl_model: str) -> Path:
    return RESULTS_DIR / f'rq2_entropy_{config}' / ssl_model


def _load_entropy_ood(config: str, ssl_model: str) -> dict:
    """Load OOD entropy results, average per (codebook, emotion, layer)."""
    results_dir = _entropy_results_dir(config, ssl_model)
    all_data = []
    for src in ID_DATASETS:
        for tgt in ID_DATASETS:
            if src == tgt:
                continue
            p = results_dir / f'{src}_to_{tgt}_ood.json'
            if p.exists():
                with open(p) as f:
                    all_data.append(json.load(f))

    if not all_data:
        return {}

    cb_names = ['balanced'] + [f'biased_{e}' for e in COMMON_EMOTIONS]
    result = {}
    for cb_name in cb_names:
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


def _compute_merged_entropy(entropy_data: dict) -> dict:
    """Compute Match and Unmatch average entropy from biased codebook data.

    Match:   For each emotion e, take biased_e[emotion=e], average across 4 emotions.
    Unmatch: For each emotion e, average biased_e[emotion!=e], then average across 4.
    """
    match_per_layer = defaultdict(list)
    unmatch_per_layer = defaultdict(list)

    for emo in COMMON_EMOTIONS:
        cb_name = f'biased_{emo}'
        cb_data = entropy_data.get(cb_name, {})

        self_vals = cb_data.get(emo, {})
        for layer, val in self_vals.items():
            match_per_layer[layer].append(val)

        other_emos = [e for e in COMMON_EMOTIONS if e != emo]
        other_per_layer = defaultdict(list)
        for other_emo in other_emos:
            other_vals = cb_data.get(other_emo, {})
            for layer, val in other_vals.items():
                other_per_layer[layer].append(val)
        for layer, vals in other_per_layer.items():
            unmatch_per_layer[layer].append(float(np.mean(vals)))

    match_avg = {l: float(np.mean(vs)) for l, vs in sorted(match_per_layer.items())}
    unmatch_avg = {l: float(np.mean(vs)) for l, vs in sorted(unmatch_per_layer.items())}

    return {'match': match_avg, 'unmatch': unmatch_avg}


# ============================================================
# Plotting
# ============================================================

def _plot_figure(target_ssl: str, output_path: Path, free_scale: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(7.0 * 6, 5.5))

    # 1 row × 6 columns: (a) SER cols 0-1, (b) Entropy cols 2-5
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 4], wspace=0.18)
    gs_ser = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.30)
    gs_ent = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.30)

    layers = list(range(1, NUM_LAYERS + 1))

    ssl_configs = [
        ('e2v', E2V_CONFIG, f'{SSL_DISPLAY["e2v"]} ({E2V_CONFIG})'),
        (target_ssl, TARGET_CONFIG, f'{SSL_DISPLAY[target_ssl]} ({TARGET_CONFIG})'),
    ]

    ser_axes = []
    ent_axes = []

    # ---- (a) SER Recall: 2 subplots ----
    for col_idx, (ssl, config, title) in enumerate(ssl_configs):
        share_ax = ser_axes[0] if (col_idx > 0 and not free_scale) else None
        ax = fig.add_subplot(gs_ser[col_idx], sharey=share_ax)
        ser_axes.append(ax)
        data = _load_ser_ood_avg(config, ssl)

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
        else:
            for ct_key, style in CODEBOOK_TYPE_STYLES.items():
                vals = data.get(ct_key, {})
                xs = sorted(vals.keys())
                ys = [vals[x] for x in xs]
                if ys:
                    ax.plot(xs, ys,
                            label=style['label'],
                            color=style['color'],
                            marker=style['marker'],
                            linestyle=style['ls'],
                            linewidth=style.get('lw', 2.5),
                            markersize=6)

            bl_val = _load_recall_baseline(ssl)
            if bl_val is not None:
                ax.axhline(y=bl_val, color='#666666',
                           linestyle=':', linewidth=1.5, alpha=0.5,
                           label='Unquantized')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('RVQ Layer', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel('SER Recall', fontsize=13)
        elif not free_scale:
            import matplotlib.pyplot as plt
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xticks(layers[::2])
        ax.set_xlim(0.5, NUM_LAYERS + 0.5)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')

    # ---- (b) Entropy: 4 subplots ----
    ent_col = 0
    for ssl, config, _title in ssl_configs:
        ssl_label = SSL_DISPLAY[ssl]
        entropy_data = _load_entropy_ood(config, ssl)

        # Balanced entropy
        ax_bal = fig.add_subplot(gs_ent[ent_col])
        ent_axes.append(ax_bal)
        if entropy_data:
            bal_data = entropy_data.get('balanced', {})
            for emo in COMMON_EMOTIONS:
                ent = bal_data.get(emo, {})
                xs = sorted(ent.keys())
                ys = [ent[x] for x in xs]
                if ys:
                    ax_bal.plot(xs, ys,
                               label=EMOTION_DISPLAY[emo],
                               color=EMOTION_COLORS[emo],
                               linestyle='-', linewidth=2.5,
                               marker='o', markersize=5)
        else:
            ax_bal.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_bal.transAxes, fontsize=14)

        ax_bal.set_title(f'Balanced ({ssl_label})', fontsize=13, fontweight='bold')
        ax_bal.set_xlabel('RVQ Layer', fontsize=11)
        if ent_col == 0:
            ax_bal.set_ylabel('Normalized Entropy', fontsize=12)
        ax_bal.set_xticks(layers[::2])
        ax_bal.set_xlim(0.5, NUM_LAYERS + 0.5)
        if not free_scale:
            ax_bal.set_ylim(-0.05, 1.05)
        ax_bal.tick_params(axis='both', labelsize=10)
        ax_bal.grid(True, alpha=0.3)
        if ent_col == 0:
            ax_bal.legend(fontsize=9, loc='lower right')

        # Merged biased entropy
        ax_merge = fig.add_subplot(gs_ent[ent_col + 1])
        ent_axes.append(ax_merge)
        if entropy_data:
            merged = _compute_merged_entropy(entropy_data)

            match_vals = merged['match']
            xs_m = sorted(match_vals.keys())
            ys_m = [match_vals[x] for x in xs_m]
            ax_merge.plot(xs_m, ys_m,
                          label='Match', color='#1E88E5',
                          linestyle='-', linewidth=2.5,
                          marker='o', markersize=5)

            unmatch_vals = merged['unmatch']
            xs_u = sorted(unmatch_vals.keys())
            ys_u = [unmatch_vals[x] for x in xs_u]
            ax_merge.plot(xs_u, ys_u,
                          label='Unmatch', color='#E53935',
                          linestyle='--', linewidth=2.5,
                          marker='s', markersize=5)
        else:
            ax_merge.text(0.5, 0.5, 'No data', ha='center', va='center',
                          transform=ax_merge.transAxes, fontsize=14)

        ax_merge.set_title(f'Emo-specific Merged ({ssl_label})', fontsize=13, fontweight='bold')
        ax_merge.set_xlabel('RVQ Layer', fontsize=11)
        ax_merge.set_xticks(layers[::2])
        ax_merge.set_xlim(0.5, NUM_LAYERS + 0.5)
        if not free_scale:
            ax_merge.set_ylim(-0.05, 1.05)
        ax_merge.tick_params(axis='both', labelsize=10)
        ax_merge.grid(True, alpha=0.3)
        ax_merge.legend(fontsize=10, loc='lower right')

        ent_col += 2

    # Section headers centered over their groups
    fig.canvas.draw()

    ser_left = ser_axes[0].get_position().x0
    ser_right = ser_axes[-1].get_position().x1
    fig.text((ser_left + ser_right) / 2, 1.03, '(a) SER Recall',
             ha='center', fontsize=16, fontweight='bold', transform=fig.transFigure)

    ent_left = ent_axes[0].get_position().x0
    ent_right = ent_axes[-1].get_position().x1
    fig.text((ent_left + ent_right) / 2, 1.03, '(b) Codebook Token Entropy',
             ha='center', fontsize=16, fontweight='bold', transform=fig.transFigure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path.with_suffix('.pdf')), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


# ============================================================
# Entry points
# ============================================================

def run(dry_run=False):
    """Generate both HuBERT and WavLM variants, plus free-scale versions."""
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for target_ssl in ['hubert', 'wavlm']:
        output = PAPER_FIGURES_DIR / f'rq2_combined_{target_ssl}.png'
        output_free = PAPER_FIGURES_DIR / f'rq2_combined_{target_ssl}_free.png'
        if dry_run:
            print(f"  [DRY RUN] Would generate {output}")
            print(f"  [DRY RUN] Would generate {output_free}")
            continue
        _plot_figure(target_ssl, output)
        _plot_figure(target_ssl, output_free, free_scale=True)


def description() -> str:
    return "RQ2 Combined: SER Recall + Entropy (e2v vs HuBERT/WavLM)"


def main():
    parser = argparse.ArgumentParser(description=description())
    parser.add_argument('--target-ssl', type=str, default=None,
                        choices=['hubert', 'wavlm'],
                        help='Generate for specific target. Default: both.')
    args = parser.parse_args()

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    targets = [args.target_ssl] if args.target_ssl else ['hubert', 'wavlm']
    for target_ssl in targets:
        output = PAPER_FIGURES_DIR / f'rq2_combined_{target_ssl}.png'
        output_free = PAPER_FIGURES_DIR / f'rq2_combined_{target_ssl}_free.png'
        _plot_figure(target_ssl, output)
        _plot_figure(target_ssl, output_free, free_scale=True)


if __name__ == '__main__':
    main()
