#!/usr/bin/env python3
"""
RQ2.2 Table -- L2 + SER-F1 for 3 SSL models at Layer 8 (128x8, OOD avg)

Generates a LaTeX table + plain text summary:
  - Rows: emotion2vec, HuBERT, WavLM
  - Columns: L2 (match/unmatch/balanced), SER F1 (match/unmatch/balanced)
  - Values: OOD average at Layer 8

Reads from:  results/rq2_2_ssl_table_128x8/{ssl}/{src}_to_{tgt}_ood.json
Outputs to:  results/paper_figures_rq/rq2_2_table.tex  (.txt, .csv)
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

RESULTS_INPUT = RESULTS_DIR / 'rq2_2_ssl_table_128x8'
TARGET_LAYER = f'layer_{NUM_LAYERS}'

SSL_MODELS = ['e2v', 'hubert', 'wavlm']
SSL_DISPLAY = {'e2v': 'emotion2vec', 'hubert': 'HuBERT', 'wavlm': 'WavLM'}
CB_TYPES = ['biased_matched', 'biased_unmatched', 'balanced']


def _load_ood_avg(ssl: str) -> dict:
    """Load all OOD JSONs for one SSL model, return avg metrics at target layer."""
    ssl_dir = RESULTS_INPUT / ssl
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
    for ct in CB_TYPES:
        l2_vals, f1_vals = [], []
        for d in all_data:
            layer_data = d.get(ct, {}).get(TARGET_LAYER, {})
            if layer_data:
                l2_vals.append(abs(layer_data.get('l2', 0.0)))
                f1_vals.append(layer_data.get('f1_macro', 0.0))
        if l2_vals:
            result[ct] = {
                'l2': float(np.mean(l2_vals)),
                'f1': float(np.mean(f1_vals)),
            }
    return result


def _generate_table(output_dir: Path):
    rows = {}
    for ssl in SSL_MODELS:
        rows[ssl] = _load_ood_avg(ssl)

    has_data = any(bool(v) for v in rows.values())
    if not has_data:
        print("  [WARNING] No OOD data found in", RESULTS_INPUT)
        print("  Run the evaluator first or wait for training to complete.")
        return

    # --- Plain text ---
    txt_path = output_dir / 'rq2_2_table.txt'
    lines = []
    header = f"{'SSL':<12} {'L2-match':>10} {'L2-unmatch':>12} {'L2-balanced':>12} {'F1-match':>10} {'F1-unmatch':>12} {'F1-balanced':>12}"
    lines.append(header)
    lines.append('-' * len(header))
    for ssl in SSL_MODELS:
        data = rows[ssl]
        if not data:
            lines.append(f"{SSL_DISPLAY[ssl]:<12} {'N/A':>10} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
            continue
        m = data.get('biased_matched', {})
        u = data.get('biased_unmatched', {})
        b = data.get('balanced', {})
        lines.append(
            f"{SSL_DISPLAY[ssl]:<12} "
            f"{m.get('l2', 0):>10.4f} {u.get('l2', 0):>12.4f} {b.get('l2', 0):>12.4f} "
            f"{m.get('f1', 0):>10.4f} {u.get('f1', 0):>12.4f} {b.get('f1', 0):>12.4f}"
        )
    txt_content = '\n'.join(lines)
    with open(txt_path, 'w') as f:
        f.write(txt_content + '\n')
    print(txt_content)
    print(f"\n  Saved: {txt_path}")

    # --- CSV ---
    csv_path = output_dir / 'rq2_2_table.csv'
    csv_lines = ['SSL,L2_match,L2_unmatch,L2_balanced,F1_match,F1_unmatch,F1_balanced']
    for ssl in SSL_MODELS:
        data = rows[ssl]
        if not data:
            csv_lines.append(f"{SSL_DISPLAY[ssl]},,,,,, ")
            continue
        m = data.get('biased_matched', {})
        u = data.get('biased_unmatched', {})
        b = data.get('balanced', {})
        csv_lines.append(
            f"{SSL_DISPLAY[ssl]},"
            f"{m.get('l2', '')},{u.get('l2', '')},{b.get('l2', '')},"
            f"{m.get('f1', '')},{u.get('f1', '')},{b.get('f1', '')}"
        )
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"  Saved: {csv_path}")

    # --- LaTeX ---
    tex_path = output_dir / 'rq2_2_table.tex'
    tex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'\multirow{2}{*}{\textbf{SSL}} & \multicolumn{3}{c}{\textbf{L2 Distance}} & \multicolumn{3}{c}{\textbf{SER F1-Macro}} \\',
        r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}',
        r' & matched & unmatched & balanced & matched & unmatched & balanced \\',
        r'\midrule',
    ]
    for ssl in SSL_MODELS:
        data = rows[ssl]
        if not data:
            tex_lines.append(f'{SSL_DISPLAY[ssl]} & -- & -- & -- & -- & -- & -- \\\\')
            continue
        m = data.get('biased_matched', {})
        u = data.get('biased_unmatched', {})
        b = data.get('balanced', {})
        tex_lines.append(
            f"{SSL_DISPLAY[ssl]} & "
            f"{m.get('l2', 0):.4f} & {u.get('l2', 0):.4f} & {b.get('l2', 0):.4f} & "
            f"{m.get('f1', 0):.4f} & {u.get('f1', 0):.4f} & {b.get('f1', 0):.4f} \\\\"
        )
    tex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{Model Evaluation Results (128$\\times$8, Layer {NUM_LAYERS}, OOD avg)}}',
        r'\label{tab:rq2_2_results}',
        r'\end{table}',
    ])
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  Saved: {tex_path}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [DRY RUN] Would generate table in {PAPER_FIGURES_DIR}")
        return
    _generate_table(PAPER_FIGURES_DIR)


def description():
    return "RQ2.2: SSL comparison table (L2 + SER-F1, 128x8, OOD)"
