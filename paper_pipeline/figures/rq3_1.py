#!/usr/bin/env python3
"""
RQ3.1 Table -- Cosine + SER-F1 for ratio codebooks across 3 SSL models

Layout:
  Rows: emotion2vec, HuBERT, WavLM
  Columns (2 groups):
    Cosine Similarity:  Balanced | 10+90 | ... | Biased
    SER F1-Macro:       Balanced | 10+90 | ... | Biased

Values: OOD average ± std at the last RVQ layer (layer 24).

Reads from:  results/rq3_1_ratio/{cb_config}/{ssl}/{src}_to_{tgt}_ood.json
Outputs to:  results/paper_figures_rq/rq3_1_table.tex  (.txt, .csv, .json)
"""

import sys
import json
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
EXP_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from config import RESULTS_DIR, PAPER_FIGURES_DIR, ID_DATASETS

NUM_LAYERS = 24
TARGET_LAYER = f'layer_{NUM_LAYERS}'

SSL_CONFIGS = [
    {'ssl': 'e2v',    'cb_config': '2x24',    'display': 'emotion2vec'},
    {'ssl': 'hubert', 'cb_config': '1024x24', 'display': 'HuBERT'},
    {'ssl': 'wavlm',  'cb_config': '1024x24', 'display': 'WavLM'},
]

RATIO_KEYS = [
    'balanced', 'mixed_r10', 'mixed_r20', 'mixed_r40', 'mixed_r50',
    'mixed_r70', 'mixed_r80', 'mixed_r95', 'mixed_r99', 'biased',
]
RATIO_DISPLAY = {
    'balanced':  'Balanced',
    'mixed_r10': '10+90',
    'mixed_r20': '20+80',
    'mixed_r40': '40+60',
    'mixed_r50': '50+50',
    'mixed_r70': '70+30',
    'mixed_r80': '80+20',
    'mixed_r95': '95+5',
    'mixed_r99': '99+1',
    'biased':    'Biased',
}


def _results_dir(ssl: str, cb_config: str) -> Path:
    return RESULTS_DIR / 'rq3_1_ratio' / cb_config / ssl


def _load_ood_stats(ssl: str, cb_config: str) -> dict:
    """Load all OOD JSONs, return mean ± std at target layer per ratio type."""
    ssl_dir = _results_dir(ssl, cb_config)
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
        cos_vals, f1_vals = [], []
        for d in all_data:
            layer_data = d.get(rk, {}).get(TARGET_LAYER, {})
            if layer_data:
                cos_vals.append(layer_data.get('cosine', 0.0))
                f1_vals.append(layer_data.get('f1_macro', 0.0))
        if cos_vals:
            result[rk] = {
                'cosine_mean': float(np.mean(cos_vals)),
                'cosine_std': float(np.std(cos_vals)),
                'f1_mean': float(np.mean(f1_vals)),
                'f1_std': float(np.std(f1_vals)),
                'n': len(cos_vals),
            }
    return result


def _fmt_val(mean, std, decimals=4):
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def _fmt_tex(mean, std, decimals=4):
    return f"${mean:.{decimals}f}_{{\\pm {std:.{decimals}f}}}$"


def _generate_table(output_dir: Path):
    rows = {}
    for cfg in SSL_CONFIGS:
        rows[cfg['ssl']] = _load_ood_stats(cfg['ssl'], cfg['cb_config'])

    has_data = any(bool(v) for v in rows.values())
    if not has_data:
        print("  [WARNING] No OOD data found. Run the evaluator first.")
        return

    n_ratios = len(RATIO_KEYS)

    # --- Plain text ---
    txt_path = output_dir / 'rq3_1_table.txt'
    col_hdrs = [RATIO_DISPLAY[r] for r in RATIO_KEYS]
    header = f"{'SSL':<12} " + " | ".join(
        [f"{'Cos-'+h:>20s}" for h in col_hdrs] + [f"{'F1-'+h:>20s}" for h in col_hdrs]
    )
    lines = [header, '-' * len(header)]
    for cfg in SSL_CONFIGS:
        data = rows[cfg['ssl']]
        if not data:
            lines.append(f"{cfg['display']:<12} " + " | ".join(['N/A'] * (n_ratios * 2)))
            continue
        cos_strs = [f"{_fmt_val(data[rk]['cosine_mean'], data[rk]['cosine_std']):>20s}"
                     if rk in data else f"{'N/A':>20s}" for rk in RATIO_KEYS]
        f1_strs = [f"{_fmt_val(data[rk]['f1_mean'], data[rk]['f1_std']):>20s}"
                    if rk in data else f"{'N/A':>20s}" for rk in RATIO_KEYS]
        lines.append(f"{cfg['display']:<12} " + " | ".join(cos_strs + f1_strs))
    txt_content = '\n'.join(lines)
    with open(txt_path, 'w') as f:
        f.write(txt_content + '\n')
    print(txt_content)
    print(f"\n  Saved: {txt_path}")

    # --- CSV ---
    csv_path = output_dir / 'rq3_1_table.csv'
    csv_header = 'SSL,' + ','.join(
        [f'Cos_{RATIO_DISPLAY[r]}_mean,Cos_{RATIO_DISPLAY[r]}_std' for r in RATIO_KEYS] +
        [f'F1_{RATIO_DISPLAY[r]}_mean,F1_{RATIO_DISPLAY[r]}_std' for r in RATIO_KEYS]
    )
    csv_lines = [csv_header]
    for cfg in SSL_CONFIGS:
        data = rows[cfg['ssl']]
        if not data:
            csv_lines.append(f"{cfg['display']}," + ',' * (n_ratios * 4 - 1))
            continue
        vals = []
        for rk in RATIO_KEYS:
            d = data.get(rk, {})
            vals.extend([f"{d.get('cosine_mean', '')}", f"{d.get('cosine_std', '')}"])
        for rk in RATIO_KEYS:
            d = data.get(rk, {})
            vals.extend([f"{d.get('f1_mean', '')}", f"{d.get('f1_std', '')}"])
        csv_lines.append(f"{cfg['display']}," + ','.join(vals))
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"  Saved: {csv_path}")

    # --- JSON ---
    json_path = output_dir / 'rq3_1_table.json'
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- LaTeX ---
    tex_path = output_dir / 'rq3_1_table.tex'
    col_spec = 'l' + 'c' * n_ratios + 'c' * n_ratios
    cos_col_range = f'2-{1 + n_ratios}'
    f1_col_range = f'{2 + n_ratios}-{1 + 2 * n_ratios}'

    tex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        r'\multirow{2}{*}{\textbf{SSL}} '
        + f'& \\multicolumn{{{n_ratios}}}{{c}}{{\\textbf{{Cosine Similarity}}}} '
        + f'& \\multicolumn{{{n_ratios}}}{{c}}{{\\textbf{{SER F1-Macro}}}} \\\\',
        f'\\cmidrule(lr){{{cos_col_range}}} \\cmidrule(lr){{{f1_col_range}}}',
        ' & ' + ' & '.join([RATIO_DISPLAY[r] for r in RATIO_KEYS] * 2) + r' \\',
        r'\midrule',
    ]
    for cfg in SSL_CONFIGS:
        data = rows[cfg['ssl']]
        if not data:
            tex_lines.append(cfg['display'] + ' & ' + ' & '.join(['--'] * (n_ratios * 2)) + r' \\')
            continue
        vals = []
        for rk in RATIO_KEYS:
            d = data.get(rk, {})
            vals.append(_fmt_tex(d.get('cosine_mean', 0), d.get('cosine_std', 0)))
        for rk in RATIO_KEYS:
            d = data.get(rk, {})
            vals.append(_fmt_tex(d.get('f1_mean', 0), d.get('f1_std', 0)))
        tex_lines.append(f"{cfg['display']} & " + ' & '.join(vals) + r' \\')
    tex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        f'\\caption{{Ratio Codebook Comparison (Layer {NUM_LAYERS}, OOD avg $\\pm$ std, n=12 pairs)}}',
        r'\label{tab:rq3_1_ratio}',
        r'\end{table}',
    ])
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  Saved: {tex_path}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [DRY RUN] Would generate RQ3.1 table in {PAPER_FIGURES_DIR}")
        return
    _generate_table(PAPER_FIGURES_DIR)


def description():
    return "RQ3.1: Ratio codebook comparison table (Cosine + SER-F1, OOD, mean±std)"


if __name__ == '__main__':
    run()
