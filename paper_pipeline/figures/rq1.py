#!/usr/bin/env python3
"""
RQ1 Figure -- Balanced Codebook: Cosine Similarity & SER Recall (OOD only, 4 fair emotions)

Layout (like fig1.png):
  1 row x 6 columns, split into two sections:
    (a) Cosine Similarity:  emotion2vec | HuBERT | WavLM
    (b) SER Recall:         emotion2vec | HuBERT | WavLM

  OOD results using balanced codebooks from 4 ID datasets (12 OOD pairs per SSL).
  3 SSL configs: e2v (2×24), HuBERT (1024×24), WavLM (1024×24).
  4 emotion lines per subplot.

Data: results/rq1_balanced/{tag}/{ssl}/{src}_to_{tgt}_ood.json  (sample-level, cosine metric)
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

from config import RESULTS_DIR, PAPER_FIGURES_DIR, ID_DATASETS

NUM_LAYERS = 24
SOURCE_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']

CONFIGS = [
    {'tag': 'e2v_2x24',       'ssl': 'e2v',    'label': 'emotion2vec'},
    {'tag': 'hubert_1024x24', 'ssl': 'hubert', 'label': 'HuBERT'},
    {'tag': 'wavlm_1024x24',  'ssl': 'wavlm',  'label': 'WavLM'},
]

FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

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

BASELINE_DIR = RESULTS_DIR / 'ssl_comparison_l2_64x8_emilia_ood'
SSL_KEY_MAP = {'e2v': 'e2v', 'hubert': 'hubert', 'wavlm': 'wavlm'}


def _load_recall_baselines() -> dict:
    """Load unquantized per-emotion recall baselines, averaged across OOD test datasets.

    Returns: {ssl: {emotion: avg_recall}}
    """
    baselines = {}
    for cfg in CONFIGS:
        ssl = cfg['ssl']
        ssl_dir = BASELINE_DIR / SSL_KEY_MAP[ssl]
        per_emo = defaultdict(list)
        for ds in ID_DATASETS:
            p = ssl_dir / f'unquantized_{ds}.json'
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            for emo, info in data.get('per_emotion', {}).items():
                if emo in FAIR_EMOTIONS:
                    per_emo[emo].append(info['recall'])
        baselines[ssl] = {e: float(np.mean(vs)) for e, vs in per_emo.items() if vs}
    return baselines


def _get_results_dir(tag, ssl):
    """Resolve the results directory (cosine metric: no suffix)."""
    return RESULTS_DIR / 'rq1_balanced' / tag / ssl


def _load_ood_sample_data(results_dir: Path):
    """Load all OOD sample-level JSONs (12 cross-dataset pairs)."""
    all_samples = []
    for src in SOURCE_DATASETS:
        for tgt in SOURCE_DATASETS:
            if src == tgt:
                continue
            p = results_dir / f'{src}_to_{tgt}_ood.json'
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                all_samples.extend(data.get('samples', []))
    return all_samples


def _compute_emotion_stats(samples, num_layers):
    """From sample-level data, compute per-emotion per-layer cosine avg and recall."""
    by_emotion = defaultdict(list)
    for s in samples:
        label = s.get('true_label')
        if label in FAIR_EMOTIONS:
            by_emotion[label].append(s)

    cosine_data = {}
    recall_data = {}

    for emo in FAIR_EMOTIONS:
        emo_samples = by_emotion.get(emo, [])
        if not emo_samples:
            continue
        cos_vals = []
        rec_vals = []
        for l in range(num_layers):
            cos_sum = 0.0
            rec_sum = 0.0
            count = 0
            for s in emo_samples:
                cosines = s.get('cosines', [])
                preds = s.get('preds', [])
                if l < len(cosines):
                    cos_sum += cosines[l]
                    if l < len(preds) and preds[l] == s['true_label']:
                        rec_sum += 1.0
                    count += 1
            cos_vals.append(cos_sum / count if count > 0 else 0.0)
            rec_vals.append(rec_sum / count if count > 0 else 0.0)
        cosine_data[emo] = cos_vals
        recall_data[emo] = rec_vals

    return cosine_data, recall_data


def _plot_figure(output_path: Path, free_scale: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_ssl = len(CONFIGS)
    fig = plt.figure(figsize=(6.0 * n_ssl * 2, 5.5))

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18)
    gs_cos = gridspec.GridSpecFromSubplotSpec(1, n_ssl, subplot_spec=gs[0], wspace=0.35)
    gs_ser = gridspec.GridSpecFromSubplotSpec(1, n_ssl, subplot_spec=gs[1], wspace=0.35)

    layers = list(range(1, NUM_LAYERS + 1))

    section_specs = [
        (gs_cos, '(a) Cosine Similarity', 'Cosine Similarity', False),
        (gs_ser, '(b) SER Recall',        'SER Recall',        False),
    ]

    all_data = {}
    for cfg in CONFIGS:
        results_dir = _get_results_dir(cfg['tag'], cfg['ssl'])
        samples = _load_ood_sample_data(results_dir)
        if samples:
            cosine_data, recall_data = _compute_emotion_stats(samples, NUM_LAYERS)
            all_data[cfg['tag']] = {'cosine': cosine_data, 'recall': recall_data}
        else:
            print(f"  [WARNING] No OOD data for {cfg['tag']} in {results_dir}")
            all_data[cfg['tag']] = None

    baselines = _load_recall_baselines()

    section_axes = {}
    for sec_gs, sec_title, ylabel, _unused in section_specs:
        data_key = 'cosine' if 'Cosine' in sec_title else 'recall'
        axes_in_section = []

        for col_idx, cfg in enumerate(CONFIGS):
            share_ax = axes_in_section[0] if (col_idx > 0 and not free_scale) else None
            ax = fig.add_subplot(sec_gs[col_idx], sharey=share_ax)
            axes_in_section.append(ax)
            tag_data = all_data.get(cfg['tag'])

            if tag_data is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
            else:
                metric_data = tag_data[data_key]
                for emo in FAIR_EMOTIONS:
                    if emo not in metric_data:
                        continue
                    ys = metric_data[emo]
                    ax.plot(layers, ys,
                            label=EMOTION_DISPLAY[emo],
                            color=EMOTION_COLORS[emo],
                            linestyle='-',
                            marker='o',
                            linewidth=2.5,
                            markersize=6)

                if data_key == 'recall':
                    ssl_bl = baselines.get(cfg['ssl'], {})
                    for emo in FAIR_EMOTIONS:
                        if emo in ssl_bl:
                            ax.axhline(y=ssl_bl[emo],
                                       color=EMOTION_COLORS[emo],
                                       linestyle=':', linewidth=1.2, alpha=0.45)

            ax.set_title(cfg['label'], fontsize=15, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=13)
            elif not free_scale:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_xlabel('RVQ Layer', fontsize=12)
            ax.set_xticks(layers[::2])
            ax.set_xlim(0.5, NUM_LAYERS + 0.5)
            ax.tick_params(axis='both', labelsize=11)
            ax.grid(True, alpha=0.3)

        section_axes[sec_title] = axes_in_section

    fig.legend(
        [plt.Line2D([0], [0], color=EMOTION_COLORS[e], marker='o', linewidth=2.5, markersize=6)
         for e in FAIR_EMOTIONS],
        [EMOTION_DISPLAY[e] for e in FAIR_EMOTIONS],
        loc='lower center', ncol=len(FAIR_EMOTIONS), fontsize=13,
        bbox_to_anchor=(0.5, -0.08), frameon=False,
    )

    fig.canvas.draw()
    for sec_title, axes_list in section_axes.items():
        left = axes_list[0].get_position().x0
        right = axes_list[-1].get_position().x1
        center_x = (left + right) / 2.0
        fig.text(center_x, 1.03, sec_title, ha='center', fontsize=16, fontweight='bold',
                 transform=fig.transFigure)

    plt.subplots_adjust(bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path.with_suffix('.pdf')), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = PAPER_FIGURES_DIR / 'rq1_balanced_ssl_ood.png'
    output_free = PAPER_FIGURES_DIR / 'rq1_balanced_ssl_ood_free.png'

    if dry_run:
        print(f"  [DRY RUN] Would generate {output}")
        print(f"  [DRY RUN] Would generate {output_free}")
        return

    _plot_figure(output)
    _plot_figure(output_free, free_scale=True)


def description() -> str:
    return "RQ1: Balanced Codebook Cosine Similarity + SER Recall (OOD, 3 SSL configs)"


if __name__ == '__main__':
    run()
