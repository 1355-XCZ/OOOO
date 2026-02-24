#!/usr/bin/env python3
"""
RQ4 Table -- Unified Method Comparison across Codebook Structures

Reads rq4_f1/{config}/{source}.json results and generates a LaTeX table.

Rows:    codebook structures (e.g., 2x16, 2x24, ...)
Columns: N (total codes), BL, BalLS, BalSel, Unfilt, Filt, FltFBLS, FltFBs, UnfID, FiltID

Values:  OOD Macro-F1 (%) averaged over 4 sources × 4 OOD datasets
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
EXP_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from config import RESULTS_DIR, PAPER_FIGURES_DIR, ID_DATASETS

RQ4_DIR = RESULTS_DIR / 'rq4_f1'

CONFIGS_ORDER = [
    '2x16', '2x24', '2x32', '2x48', '2x64', '2x128', '2x256',
    '4x16', '4x32',
    '8x24',
    '16x16', '16x24', '16x32',
    '32x8',
    '64x8',
    '128x8', '128x12', '128x16', '128x24',
    '256x12',
    '512x16',
    '1024x16',
    '2048x16',
]

METHOD_MAP = [
    ('BL',      'Baseline'),
    ('BalLS',   'Bal_LS'),
    ('BalSel',  'Bal_Select'),
    ('Unfilt',  'Unfilt'),
    ('Filt',    'Filt'),
    ('FltFBLS', 'Filt_FBLS'),
    ('FltFBs',  'Filt_FBsel'),
    ('UnfID',   'Unfilt_ID'),
    ('FiltID',  'Filt_ID'),
]


def _total_codes(config: str) -> int:
    c, l = config.split('x')
    return int(c) * int(l)


def _load_config_avg_f1(config: str) -> Dict[str, float]:
    """Load and average F1 across 4 sources for one config."""
    method_vals = {display: [] for display, _ in METHOD_MAP}

    for source in ID_DATASETS:
        p = RQ4_DIR / config / f'{source}.json'
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for display, internal in METHOD_MAP:
            avg_f1 = data.get(internal, {}).get('ood_avg_f1', None)
            if avg_f1 is not None:
                method_vals[display].append(avg_f1)

    result = {}
    for display, _ in METHOD_MAP:
        vals = method_vals[display]
        result[display] = float(np.mean(vals)) * 100 if vals else None
    return result


def _collect_all() -> Dict[str, Dict[str, float]]:
    """Collect table data: {config: {method_display: f1_pct}}."""
    table = {}
    for config in CONFIGS_ORDER:
        row = _load_config_avg_f1(config)
        if any(v is not None for v in row.values()):
            table[config] = row
    return table


def _find_col_best(table: Dict) -> Dict[str, float]:
    """Find the best value per column for bolding."""
    best = {}
    for display, _ in METHOD_MAP:
        vals = [table[c][display] for c in table if table[c][display] is not None]
        best[display] = max(vals) if vals else 0
    return best


def _write_txt(table: Dict, path: Path):
    col_best = _find_col_best(table)
    header_cols = ['Codebook', 'N'] + [d for d, _ in METHOD_MAP]
    widths = [max(10, len(h) + 2) for h in header_cols]

    lines = ['  '.join(h.rjust(w) for h, w in zip(header_cols, widths))]
    lines.append('-' * sum(widths + [2 * (len(widths) - 1)]))

    for config in CONFIGS_ORDER:
        if config not in table:
            continue
        row = table[config]
        n = _total_codes(config)
        cells = [config.rjust(widths[0]), str(n).rjust(widths[1])]
        for i, (display, _) in enumerate(METHOD_MAP):
            v = row[display]
            cells.append(f'{v:.2f}'.rjust(widths[i + 2]) if v is not None else '--'.rjust(widths[i + 2]))
        lines.append('  '.join(cells))

    txt = '\n'.join(lines)
    path.write_text(txt)
    print(txt)
    print(f"\n  Saved: {path}")


def _write_csv(table: Dict, path: Path):
    header = ['Codebook', 'N'] + [d for d, _ in METHOD_MAP]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for config in CONFIGS_ORDER:
            if config not in table:
                continue
            row = table[config]
            n = _total_codes(config)
            cells = [config, n]
            for display, _ in METHOD_MAP:
                v = row[display]
                cells.append(f'{v:.2f}' if v is not None else '')
            w.writerow(cells)
    print(f"  Saved: {path}")


def _write_latex(table: Dict, path: Path):
    col_best = _find_col_best(table)
    n_data_cols = len(METHOD_MAP)
    col_spec = 'lc' + 'c' * n_data_cols

    lines = [
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
    ]

    header_cells = ['\\textbf{Codebook}', '\\textbf{N}']
    for display, _ in METHOD_MAP:
        header_cells.append(f'\\textbf{{{display}}}')
    lines.append(' & '.join(header_cells) + ' \\\\')
    lines.append('\\midrule')

    for config in CONFIGS_ORDER:
        if config not in table:
            continue
        row = table[config]
        n = _total_codes(config)
        cells = [config.replace('x', '$\\times$'), str(n)]
        for display, _ in METHOD_MAP:
            v = row[display]
            if v is None:
                cells.append('--')
            elif abs(v - col_best.get(display, -1)) < 0.005:
                cells.append(f'\\textbf{{{v:.2f}}}')
            else:
                cells.append(f'{v:.2f}')
        lines.append(' & '.join(cells) + ' \\\\')

    lines.extend(['\\bottomrule', '\\end{tabular}'])
    tex = '\n'.join(lines)
    path.write_text(tex)
    print(f"  Saved: {path}")


def _write_json(table: Dict, path: Path):
    out = {}
    for config in CONFIGS_ORDER:
        if config not in table:
            continue
        out[config] = {'N': _total_codes(config), **table[config]}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


def _build_delta_table(table: Dict) -> Dict[str, Dict[str, float]]:
    """Compute delta = method - BL for each cell (skip BL column itself)."""
    delta = {}
    for config, row in table.items():
        bl = row.get('BL')
        if bl is None:
            continue
        delta[config] = {}
        for display, _ in METHOD_MAP:
            v = row[display]
            if v is None or display == 'BL':
                delta[config][display] = None
            else:
                delta[config][display] = v - bl
    return delta


DELTA_METHODS = [(d, i) for d, i in METHOD_MAP if d != 'BL']


def _write_delta_txt(delta: Dict, path: Path):
    header_cols = ['Codebook', 'N'] + [d for d, _ in DELTA_METHODS]
    widths = [max(10, len(h) + 2) for h in header_cols]

    lines = ['  '.join(h.rjust(w) for h, w in zip(header_cols, widths))]
    lines.append('-' * sum(widths + [2 * (len(widths) - 1)]))

    for config in CONFIGS_ORDER:
        if config not in delta:
            continue
        row = delta[config]
        n = _total_codes(config)
        cells = [config.rjust(widths[0]), str(n).rjust(widths[1])]
        for i, (display, _) in enumerate(DELTA_METHODS):
            v = row.get(display)
            if v is None:
                cells.append('--'.rjust(widths[i + 2]))
            else:
                s = f'{v:+.2f}'
                cells.append(s.rjust(widths[i + 2]))
        lines.append('  '.join(cells))

    txt = '\n'.join(lines)
    path.write_text(txt)
    print(txt)
    print(f"\n  Saved: {path}")


def _write_delta_csv(delta: Dict, path: Path):
    header = ['Codebook', 'N'] + [d for d, _ in DELTA_METHODS]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for config in CONFIGS_ORDER:
            if config not in delta:
                continue
            row = delta[config]
            n = _total_codes(config)
            cells = [config, n]
            for display, _ in DELTA_METHODS:
                v = row.get(display)
                cells.append(f'{v:+.2f}' if v is not None else '')
            w.writerow(cells)
    print(f"  Saved: {path}")


def _write_delta_latex(delta: Dict, path: Path):
    col_best = {}
    col_worst = {}
    for display, _ in DELTA_METHODS:
        vals = [delta[c][display] for c in delta if delta[c].get(display) is not None]
        col_best[display] = max(vals) if vals else 0
        col_worst[display] = min(vals) if vals else 0

    n_data_cols = len(DELTA_METHODS)
    col_spec = 'lc' + 'c' * n_data_cols

    lines = [
        f'\\begin{{tabular}}{{{col_spec}}}',
        '\\toprule',
    ]

    header_cells = ['\\textbf{Codebook}', '\\textbf{N}']
    for display, _ in DELTA_METHODS:
        header_cells.append(f'\\textbf{{$\\Delta${display}}}')
    lines.append(' & '.join(header_cells) + ' \\\\')
    lines.append('\\midrule')

    for config in CONFIGS_ORDER:
        if config not in delta:
            continue
        row = delta[config]
        n = _total_codes(config)
        cells = [config.replace('x', '$\\times$'), str(n)]
        for display, _ in DELTA_METHODS:
            v = row.get(display)
            if v is None:
                cells.append('--')
            else:
                s = f'{v:+.2f}'
                if abs(v - col_best[display]) < 0.005:
                    cells.append(f'\\textbf{{{s}}}')
                else:
                    cells.append(s)
        lines.append(' & '.join(cells) + ' \\\\')

    lines.extend(['\\bottomrule', '\\end{tabular}'])
    tex = '\n'.join(lines)
    path.write_text(tex)
    print(f"  Saved: {path}")


def _write_delta_json(delta: Dict, path: Path):
    out = {}
    for config in CONFIGS_ORDER:
        if config not in delta:
            continue
        out[config] = {'N': _total_codes(config)}
        for display, _ in DELTA_METHODS:
            out[config][display] = delta[config].get(display)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


def run(dry_run=False):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("  [DRY RUN] Would generate rq4_table.{txt,csv,tex,json} + rq4_delta.{...}")
        return

    table = _collect_all()
    if not table:
        print("  [WARNING] No RQ4 F1 data found. Run evaluations first.")
        return

    base = PAPER_FIGURES_DIR / 'rq4_table'
    _write_txt(table, base.with_suffix('.txt'))
    _write_csv(table, base.with_suffix('.csv'))
    _write_latex(table, base.with_suffix('.tex'))
    _write_json(table, base.with_suffix('.json'))

    delta = _build_delta_table(table)
    dbase = PAPER_FIGURES_DIR / 'rq4_delta'
    print()
    _write_delta_txt(delta, dbase.with_suffix('.txt'))
    _write_delta_csv(delta, dbase.with_suffix('.csv'))
    _write_delta_latex(delta, dbase.with_suffix('.tex'))
    _write_delta_json(delta, dbase.with_suffix('.json'))


def description():
    return "RQ4: Unified Method Comparison across Codebook Structures (OOD Macro-F1)"


if __name__ == '__main__':
    run()
