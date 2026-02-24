#!/usr/bin/env python3
"""
RQ3.2 Table -- Cosine + SER-F1 for ambiguity codebooks across 3 SSL models

Layout:
  Rows: emotion2vec, HuBERT, WavLM
  Columns (2 groups):
    Cosine Similarity:  Low (0%) | Mid (20%) | High (33%)
    SER F1-Macro:       Low (0%) | Mid (20%) | High (33%)

Values: OOD average ± std at the last RVQ layer (layer 24).
Codebook source: IEMOCAP only.
OOD targets: esd_en, ravdess, cremad (3 pairs).

Reads from:  results/rq3_2_ambiguity/{cb_config}/{ssl}/iemocap_to_{tgt}_ood.json
Outputs to:  results/paper_figures_rq/rq3_2_table.tex  (.txt, .csv, .json)
"""

import sys
import json
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
EXP_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from config import RESULTS_DIR, PAPER_FIGURES_DIR

NUM_LAYERS = 24
TARGET_LAYER = f'layer_{NUM_LAYERS}'
CODEBOOK_DATASET = 'iemocap'
OOD_TARGETS = ['esd_en', 'ravdess', 'cremad']

SSL_CONFIGS = [
    {'ssl': 'e2v',    'cb_config': '2x24',    'display': 'emotion2vec'},
    {'ssl': 'hubert', 'cb_config': '1024x24', 'display': 'HuBERT'},
    {'ssl': 'wavlm',  'cb_config': '1024x24', 'display': 'WavLM'},
]

AMBIGUITY_KEYS = ['high', 'mid', 'low']
AMBIGUITY_DISPLAY = {
    'high': 'Low (0\\%)',
    'mid':  'Mid (20\\%)',
    'low':  'High (33\\%)',
}
AMBIGUITY_DISPLAY_PLAIN = {
    'high': 'Low(0%)',
    'mid':  'Mid(20%)',
    'low':  'High(33%)',
}


def _results_dir(ssl: str, cb_config: str) -> Path:
    return RESULTS_DIR / 'rq3_2_ambiguity' / cb_config / ssl


def _load_ood_stats(ssl: str, cb_config: str) -> dict:
    """Load all OOD JSONs, return mean ± std at target layer per ambiguity level."""
    ssl_dir = _results_dir(ssl, cb_config)
    all_data = []
    for tgt in OOD_TARGETS:
        p = ssl_dir / f'{CODEBOOK_DATASET}_to_{tgt}_ood.json'
        if p.exists():
            with open(p) as f:
                all_data.append(json.load(f))

    if not all_data:
        return {}

    result = {}
    for ak in AMBIGUITY_KEYS:
        cos_vals, f1_vals = [], []
        for d in all_data:
            layer_data = d.get(ak, {}).get(TARGET_LAYER, {})
            if layer_data:
                cos_vals.append(layer_data.get('cosine', 0.0))
                f1_vals.append(layer_data.get('f1_macro', 0.0))
        if cos_vals:
            result[ak] = {
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

    n_levels = len(AMBIGUITY_KEYS)

    # --- Plain text ---
    txt_path = output_dir / 'rq3_2_table.txt'
    col_hdrs = [AMBIGUITY_DISPLAY_PLAIN[a] for a in AMBIGUITY_KEYS]
    header = f"{'SSL':<12} " + " | ".join(
        [f"{'Cos-'+h:>24s}" for h in col_hdrs] + [f"{'F1-'+h:>24s}" for h in col_hdrs]
    )
    lines = [header, '-' * len(header)]
    for cfg in SSL_CONFIGS:
        data = rows[cfg['ssl']]
        if not data:
            lines.append(f"{cfg['display']:<12} " + " | ".join(['N/A'] * (n_levels * 2)))
            continue
        cos_strs = [f"{_fmt_val(data[ak]['cosine_mean'], data[ak]['cosine_std']):>24s}"
                     if ak in data else f"{'N/A':>24s}" for ak in AMBIGUITY_KEYS]
        f1_strs = [f"{_fmt_val(data[ak]['f1_mean'], data[ak]['f1_std']):>24s}"
                    if ak in data else f"{'N/A':>24s}" for ak in AMBIGUITY_KEYS]
        lines.append(f"{cfg['display']:<12} " + " | ".join(cos_strs + f1_strs))
    txt_content = '\n'.join(lines)
    with open(txt_path, 'w') as f:
        f.write(txt_content + '\n')
    print(txt_content)
    print(f"\n  Saved: {txt_path}")

    # --- CSV ---
    csv_path = output_dir / 'rq3_2_table.csv'
    csv_header = 'SSL,' + ','.join(
        [f'Cos_{AMBIGUITY_DISPLAY_PLAIN[a]}_mean,Cos_{AMBIGUITY_DISPLAY_PLAIN[a]}_std' for a in AMBIGUITY_KEYS] +
        [f'F1_{AMBIGUITY_DISPLAY_PLAIN[a]}_mean,F1_{AMBIGUITY_DISPLAY_PLAIN[a]}_std' for a in AMBIGUITY_KEYS]
    )
    csv_lines = [csv_header]
    for cfg in SSL_CONFIGS:
        data = rows[cfg['ssl']]
        if not data:
            csv_lines.append(f"{cfg['display']}," + ',' * (n_levels * 4 - 1))
            continue
        vals = []
        for ak in AMBIGUITY_KEYS:
            d = data.get(ak, {})
            vals.extend([f"{d.get('cosine_mean', '')}", f"{d.get('cosine_std', '')}"])
        for ak in AMBIGUITY_KEYS:
            d = data.get(ak, {})
            vals.extend([f"{d.get('f1_mean', '')}", f"{d.get('f1_std', '')}"])
        csv_lines.append(f"{cfg['display']}," + ','.join(vals))
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"  Saved: {csv_path}")

    # --- JSON ---
    json_path = output_dir / 'rq3_2_table.json'
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- LaTeX ---
    tex_path = output_dir / 'rq3_2_table.tex'
    col_spec = 'l' + 'c' * n_levels + 'c' * n_levels
    cos_col_range = f'2-{1 + n_levels}'
    f1_col_range = f'{2 + n_levels}-{1 + 2 * n_levels}'

    tex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        r'\multirow{2}{*}{\textbf{SSL}} '
        + f'& \\multicolumn{{{n_levels}}}{{c}}{{\\textbf{{Cosine Similarity}}}} '
        + f'& \\multicolumn{{{n_levels}}}{{c}}{{\\textbf{{SER F1-Macro}}}} \\\\',
        f'\\cmidrule(lr){{{cos_col_range}}} \\cmidrule(lr){{{f1_col_range}}}',
        ' & ' + ' & '.join([AMBIGUITY_DISPLAY[a] for a in AMBIGUITY_KEYS] * 2) + r' \\',
        r'\midrule',
    ]
    for cfg in SSL_CONFIGS:
        data = rows[cfg['ssl']]
        if not data:
            tex_lines.append(cfg['display'] + ' & ' + ' & '.join(['--'] * (n_levels * 2)) + r' \\')
            continue
        vals = []
        for ak in AMBIGUITY_KEYS:
            d = data.get(ak, {})
            vals.append(_fmt_tex(d.get('cosine_mean', 0), d.get('cosine_std', 0)))
        for ak in AMBIGUITY_KEYS:
            d = data.get(ak, {})
            vals.append(_fmt_tex(d.get('f1_mean', 0), d.get('f1_std', 0)))
        tex_lines.append(f"{cfg['display']} & " + ' & '.join(vals) + r' \\')
    tex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{Ambiguity Level Comparison (IEMOCAP codebooks, Layer {NUM_LAYERS}, OOD avg $\\pm$ std, n=3 pairs)}}',
        r'\label{tab:rq3_2_ambiguity}',
        r'\end{table}',
    ])
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  Saved: {tex_path}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [DRY RUN] Would generate RQ3.2 table in {PAPER_FIGURES_DIR}")
        return
    _generate_table(PAPER_FIGURES_DIR)


def description():
    return "RQ3.2: Ambiguity codebook comparison table (Cosine + SER-F1, IEMOCAP, OOD, mean±std)"


if __name__ == '__main__':
    run()
