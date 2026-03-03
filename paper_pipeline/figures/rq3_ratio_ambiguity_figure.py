#!/usr/bin/env python3
"""
RQ3 Ratio × Ambiguity Figure (supervisor layout)

Left:  JS divergence — 2 subplots (e2v, HuBERT), each with low & high on same axes
Right: Top2-SetAcc  — 2×2 grid (e2v top, HuBERT bottom, low left, high right)

Ambiguity: high = primary ≤ 50% (old high + mid merged), low = primary > 50%
Ratios:    balanced, 95+5, 99+1, 100+0 (biased), no-quantization

Data:  results/rq2_ce/{ssl_dir}/{source}_samples_va.json
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

from config import RESULTS_DIR, PAPER_FIGURES_DIR

SOURCES = ['esd_en', 'ravdess', 'cremad']
NUM_LAYERS = 24
FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
EPSILON = 1e-8

SSL_CONFIGS = [
    ('e2v_native',    'emotion2vec'),
    ('hubert_native', 'HuBERT'),
]

CODEBOOK_STYLES = {
    'balanced':  {'color': '#2E7D32', 'label': 'Balanced'},
    'mixed_r95': {'color': '#FBC02D', 'label': '95%+5%'},
    'mixed_r99': {'color': '#F57C00', 'label': '99%+1%'},
    'biased':    {'color': '#C62828', 'label': '100%+0%'},
    'baseline':  {'color': '#000000', 'label': 'No quantization'},
}

LOW_STYLE  = dict(linestyle='-',  marker='o', markerfmt='circle')
HIGH_STYLE = dict(linestyle='--', marker='D', markerfmt='diamond')


def _epsilon_smooth(dist, eps=EPSILON):
    smoothed = np.maximum(dist, eps)
    return smoothed / smoothed.sum()


def _kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def js_divergence(y, p, eps=EPSILON):
    y_s = _epsilon_smooth(y, eps)
    p_s = _epsilon_smooth(p, eps)
    m = 0.5 * (y_s + p_s)
    return 0.5 * _kl(y_s, m) + 0.5 * _kl(p_s, m)


def _record_js(r):
    y = np.array([r['y'][e] for e in FAIR_EMOTIONS])
    p = np.array([r['p'][e] for e in FAIR_EMOTIONS])
    return js_divergence(y, p)


def _top2_match(r):
    y = np.array([r['y'][e] for e in FAIR_EMOTIONS])
    p = np.array([r['p'][e] for e in FAIR_EMOTIONS])
    return 1 if set(np.argsort(y)[-2:]) == set(np.argsort(p)[-2:]) else 0


def _consistency(votes: dict) -> float:
    total = sum(votes.values())
    if total == 0:
        return 0.0
    return max(votes.values()) / total


def _load_samples(ssl_dir: str, version: str):
    all_records = []
    ce_dir = RESULTS_DIR / 'rq2_ce' / ssl_dir
    for src in SOURCES:
        path = ce_dir / f'{src}_samples_{version}.json'
        if not path.exists():
            print(f'  WARNING: missing {path}')
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            r['_source'] = src
        all_records.extend(records)
        print(f'  Loaded {len(records)} records from {src} ({ssl_dir})')
    return all_records


def _bin_samples_binary(records):
    """Binary split: high (primary ≤ 50%) vs low (primary > 50%)."""
    baselines = [r for r in records if r['codebook'] == 'baseline']
    seen = {}
    for r in baselines:
        uid = r['utt_id']
        if uid not in seen:
            seen[uid] = r['votes']
    bins = {'high': set(), 'low': set()}
    for uid, votes in seen.items():
        c = _consistency(votes)
        if c <= 0.5001:
            bins['high'].add(uid)
        else:
            bins['low'].add(uid)
    print(f'  Ambiguity bins: high={len(bins["high"])}, low={len(bins["low"])}')
    return bins


def _avg_metric_per_layer(records, utt_ids, codebook_type, metric_fn):
    sample_layer = defaultdict(list)
    for r in records:
        if r['utt_id'] not in utt_ids:
            continue
        cb = r['codebook']
        if codebook_type == 'balanced' and cb != 'balanced':
            continue
        if codebook_type == 'biased':
            if not cb.startswith('biased_') or cb != f"biased_{r['primary']}":
                continue
        if codebook_type.startswith('mixed_r'):
            ratio_tag = codebook_type
            expected = f"{ratio_tag}_{r['primary']}"
            if cb != expected:
                continue
        sample_layer[(r['utt_id'], r['layer'])].append(metric_fn(r))

    layer_avg = defaultdict(list)
    for (uid, layer), vals in sample_layer.items():
        if layer > 0:
            layer_avg[layer].append(float(np.mean(vals)))
    layers = sorted(layer_avg.keys())
    avg = [float(np.mean(layer_avg[l])) for l in layers]
    return layers, avg


def _baseline_metric(records, utt_ids, metric_fn):
    sample_vals = defaultdict(list)
    for r in records:
        if r['utt_id'] not in utt_ids:
            continue
        if r['codebook'] == 'baseline' and r['layer'] == 0:
            sample_vals[r['utt_id']].append(metric_fn(r))
    if not sample_vals:
        return None
    return float(np.mean([float(np.mean(v)) for v in sample_vals.values()]))


def plot_figure(version: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print('Loading data...')
    ssl_data = {}
    ssl_bins = {}
    for ssl_dir, ssl_label in SSL_CONFIGS:
        recs = _load_samples(ssl_dir, version)
        ssl_data[ssl_dir] = recs
        ssl_bins[ssl_dir] = _bin_samples_binary(recs)

    cb_types = ['balanced', 'mixed_r95', 'mixed_r99', 'biased']

    # ---- Left: JS divergence (1 row × 2 cols) ----
    # Height matches Top2 panel (2 rows × 2 cols, figsize height=6) so Y-axis pixels align
    fig_js, axes_js = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    for col, (ssl_dir, ssl_label) in enumerate(SSL_CONFIGS):
        ax = axes_js[col]
        recs = ssl_data[ssl_dir]
        bins = ssl_bins[ssl_dir]

        for amb_key, amb_sty, amb_label_suffix in [
            ('low',  LOW_STYLE,  'low'),
            ('high', HIGH_STYLE, 'high'),
        ]:
            utt_ids = bins[amb_key]

            bl = _baseline_metric(recs, utt_ids, _record_js)
            if bl is not None:
                sty = CODEBOOK_STYLES['baseline']
                ax.axhline(y=bl, color=sty['color'],
                           linestyle=amb_sty['linestyle'], linewidth=1.5,
                           alpha=0.4, label=f"{sty['label']} ({amb_label_suffix})")

            for cb_type in cb_types:
                sty = CODEBOOK_STYLES[cb_type]
                layers, vals = _avg_metric_per_layer(recs, utt_ids, cb_type, _record_js)
                if layers:
                    ax.plot(layers, vals,
                            color=sty['color'],
                            linestyle=amb_sty['linestyle'],
                            marker=amb_sty['marker'],
                            markersize=4, linewidth=1.5, alpha=0.85,
                            label=f"{sty['label']} ({amb_label_suffix})")

        ax.set_title(ssl_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('RVQ Layer', fontsize=10)
        ax.set_xlim(0.5, NUM_LAYERS + 0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        if col == 0:
            ax.set_ylabel('JS Divergence', fontsize=10)

    handles, labels = axes_js[0].get_legend_handles_labels()
    fig_js.legend(handles, labels, loc='lower center',
                  ncol=5, fontsize=7.5, frameon=True, fancybox=True,
                  bbox_to_anchor=(0.5, -0.08))
    fig_js.tight_layout(rect=[0, 0.08, 1, 1])

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        out = PAPER_FIGURES_DIR / f'rq3_js_ratio_ambiguity.{ext}'
        fig_js.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig_js)

    # ---- Right: Top2-SetAcc (2 rows × 2 cols) ----
    fig_t2, axes_t2 = plt.subplots(2, 2, figsize=(10, 6))

    for row, (ssl_dir, ssl_label) in enumerate(SSL_CONFIGS):
        for col, (amb_key, amb_label) in enumerate([('low', 'Low Amb'), ('high', 'High Amb')]):
            ax = axes_t2[row, col]
            recs = ssl_data[ssl_dir]
            utt_ids = ssl_bins[ssl_dir][amb_key]
            n = len(utt_ids)

            bl = _baseline_metric(recs, utt_ids, _top2_match)
            if bl is not None:
                sty = CODEBOOK_STYLES['baseline']
                ax.axhline(y=bl, color=sty['color'], linestyle=':', linewidth=1.5,
                           alpha=0.4, label=sty['label'])

            for cb_type in cb_types:
                sty = CODEBOOK_STYLES[cb_type]
                layers, vals = _avg_metric_per_layer(recs, utt_ids, cb_type, _top2_match)
                if layers:
                    ax.plot(layers, vals,
                            color=sty['color'], marker='o', markersize=4,
                            linewidth=1.5, alpha=0.85, label=sty['label'])

            ax.set_title(f'{ssl_label} — {amb_label} (n={n})', fontsize=10, fontweight='bold')
            ax.set_xlabel('RVQ Layer', fontsize=9)
            ax.set_xlim(0.5, NUM_LAYERS + 0.5)
            ax.grid(True, alpha=0.3, linestyle='--')

            if col == 0:
                ax.set_ylabel('Top-2 Set Accuracy', fontsize=10)

        ymin = min(axes_t2[row, 0].get_ylim()[0], axes_t2[row, 1].get_ylim()[0])
        ymax = max(axes_t2[row, 0].get_ylim()[1], axes_t2[row, 1].get_ylim()[1])
        axes_t2[row, 0].set_ylim(ymin, ymax)
        axes_t2[row, 1].set_ylim(ymin, ymax)
        plt.setp(axes_t2[row, 1].get_yticklabels(), visible=False)

    handles, labels = axes_t2[0, 0].get_legend_handles_labels()
    fig_t2.legend(handles, labels, loc='lower center',
                  ncol=len(labels), fontsize=8, frameon=True, fancybox=True,
                  bbox_to_anchor=(0.5, -0.04))
    fig_t2.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ['png', 'pdf']:
        out = PAPER_FIGURES_DIR / f'rq3_top2_ratio_ambiguity.{ext}'
        fig_t2.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig_t2)

    # ---- Combined: JS (left) + Top2 (right) in one figure ----
    fig = plt.figure(figsize=(20, 4.5))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           width_ratios=[1, 1, 1, 1],
                           height_ratios=[1, 1],
                           hspace=0.45, wspace=0.28)

    # Left: JS (spans both rows, 2 cols)
    ax_js = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])]
    for col, (ssl_dir, ssl_label) in enumerate(SSL_CONFIGS):
        ax = ax_js[col]
        recs = ssl_data[ssl_dir]
        bins = ssl_bins[ssl_dir]
        for amb_key, amb_sty, amb_sfx in [('low', LOW_STYLE, 'low'), ('high', HIGH_STYLE, 'high')]:
            utt_ids = bins[amb_key]
            bl = _baseline_metric(recs, utt_ids, _record_js)
            if bl is not None:
                s = CODEBOOK_STYLES['baseline']
                ax.axhline(y=bl, color=s['color'], linestyle=amb_sty['linestyle'],
                           linewidth=1.5, alpha=0.4, label=f"{s['label']} ({amb_sfx})")
            for cb_type in cb_types:
                s = CODEBOOK_STYLES[cb_type]
                layers, vals = _avg_metric_per_layer(recs, utt_ids, cb_type, _record_js)
                if layers:
                    ax.plot(layers, vals, color=s['color'], linestyle=amb_sty['linestyle'],
                            marker=amb_sty['marker'], markersize=4, linewidth=1.5, alpha=0.85,
                            label=f"{s['label']} ({amb_sfx})")
        ax.set_title(ssl_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('RVQ Layer', fontsize=10)
        ax.set_xlim(0.5, NUM_LAYERS + 0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        if col == 0:
            ax.set_ylabel('JS Divergence', fontsize=10)
    js_ylims = [ax.get_ylim() for ax in ax_js]
    ymin = min(lo for lo, _ in js_ylims)
    ymax = max(hi for _, hi in js_ylims)
    for ax in ax_js:
        ax.set_ylim(ymin, ymax)
    plt.setp(ax_js[1].get_yticklabels(), visible=False)

    # Right: Top2 (2 rows × 2 cols)
    ax_t2 = [[fig.add_subplot(gs[r, c + 2]) for c in range(2)] for r in range(2)]
    for row, (ssl_dir, ssl_label) in enumerate(SSL_CONFIGS):
        for col, (amb_key, amb_label) in enumerate([('low', 'Low Amb'), ('high', 'High Amb')]):
            ax = ax_t2[row][col]
            recs = ssl_data[ssl_dir]
            utt_ids = ssl_bins[ssl_dir][amb_key]
            n = len(utt_ids)
            bl = _baseline_metric(recs, utt_ids, _top2_match)
            if bl is not None:
                s = CODEBOOK_STYLES['baseline']
                ax.axhline(y=bl, color=s['color'], linestyle=':', linewidth=1.5,
                           alpha=0.4, label=s['label'])
            for cb_type in cb_types:
                s = CODEBOOK_STYLES[cb_type]
                layers, vals = _avg_metric_per_layer(recs, utt_ids, cb_type, _top2_match)
                if layers:
                    ax.plot(layers, vals, color=s['color'], marker='o', markersize=4,
                            linewidth=1.5, alpha=0.85, label=s['label'])
            ax.set_title(f'{ssl_label} — {amb_label} (n={n})', fontsize=9, fontweight='bold')
            ax.set_xlabel('RVQ Layer', fontsize=9)
            ax.set_xlim(0.5, NUM_LAYERS + 0.5)
            ax.grid(True, alpha=0.3, linestyle='--')
            if col == 0:
                ax.set_ylabel('Top-2 Set Acc', fontsize=10)
        row_ymin = min(ax_t2[row][0].get_ylim()[0], ax_t2[row][1].get_ylim()[0])
        row_ymax = max(ax_t2[row][0].get_ylim()[1], ax_t2[row][1].get_ylim()[1])
        ax_t2[row][0].set_ylim(row_ymin, row_ymax)
        ax_t2[row][1].set_ylim(row_ymin, row_ymax)
        plt.setp(ax_t2[row][1].get_yticklabels(), visible=False)

    # Unified legend: JS handles have (low)/(high) line style info
    handles_js, labels_js = ax_js[0].get_legend_handles_labels()
    fig.legend(handles_js, labels_js, loc='lower center',
               ncol=10, fontsize=6.5, frameon=True, fancybox=True,
               columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.5, -0.08))

    for ext in ['png', 'pdf']:
        out = PAPER_FIGURES_DIR / f'rq3_combined_js_top2.{ext}'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)

    print('Done.')


def run(dry_run=False):
    if dry_run:
        out = PAPER_FIGURES_DIR / 'rq3_combined_js_top2.png'
        print(f"  [DRY RUN] Would generate {out}")
        return
    plot_figure('va')


def main():
    parser = argparse.ArgumentParser(description='RQ3 Ratio × Ambiguity Figure')
    parser.add_argument('--version', type=str, default='va', choices=['va', 'vb'])
    args = parser.parse_args()
    plot_figure(args.version)


if __name__ == '__main__':
    main()
