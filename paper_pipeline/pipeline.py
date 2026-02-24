#!/usr/bin/env python3
"""
Paper Pipeline -- RQ-based Orchestrator

Runs evaluations and figure generation organized by Research Question.

Usage:
    python -m paper_pipeline.pipeline --list
    python -m paper_pipeline.pipeline --rq 2.1
    python -m paper_pipeline.pipeline --rq 2.1 --eval
    python -m paper_pipeline.pipeline --rq 2.1 --plot
    python -m paper_pipeline.pipeline --rq all
    python -m paper_pipeline.pipeline --rq all --dry-run
"""

import sys
import argparse
import importlib
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
EXP_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(PIPELINE_DIR))

from config import RQ_REGISTRY


def _load_module(module_path: str):
    """Dynamically import a module relative to paper_pipeline package."""
    return importlib.import_module(f'paper_pipeline.{module_path}')


def list_rqs():
    print(f"\n{'RQ':<8} {'Description':<65} {'Eval':>6} {'Plot':>6}")
    print('-' * 90)
    for rq_id, info in RQ_REGISTRY.items():
        has_eval = 'yes' if info['evaluator'] else '-'
        has_plot = 'yes' if info['figure'] else '-'
        print(f"  {rq_id:<6} {info['description']:<65} {has_eval:>6} {has_plot:>6}")
    print()


def run_rq(rq_id: str, eval_only: bool = False, plot_only: bool = False,
           dry_run: bool = False):
    """Run evaluation and/or figure generation for one RQ."""
    info = RQ_REGISTRY[rq_id]

    print(f"\n{'=' * 60}")
    print(f"  RQ {rq_id}: {info['description']}")
    print(f"{'=' * 60}\n")

    run_eval = not plot_only
    run_plot = not eval_only

    if run_eval and info['evaluator']:
        print(f"  [EVAL] Loading {info['evaluator']}...")
        mod = _load_module(info['evaluator'])
        if hasattr(mod, 'run'):
            mod.run(dry_run=dry_run)
        else:
            print(f"  [SKIP] {info['evaluator']}: no run() defined")

    if run_plot and info['figure']:
        print(f"  [PLOT] Loading {info['figure']}...")
        mod = _load_module(info['figure'])
        if hasattr(mod, 'run'):
            mod.run(dry_run=dry_run)
        else:
            print(f"  [SKIP] {info['figure']}: no run() defined")


def main():
    parser = argparse.ArgumentParser(
        description='Paper Pipeline -- RQ-based orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                 List all Research Questions
  %(prog)s --rq 2.1              Run eval + plot for RQ2.1
  %(prog)s --rq 2.1 --eval       Run only evaluation for RQ2.1
  %(prog)s --rq 2.1 --plot       Run only figure generation for RQ2.1
  %(prog)s --rq all              Run all implemented RQs
  %(prog)s --rq all --dry-run    Dry-run everything
""",
    )
    parser.add_argument('--list', action='store_true', help='List all RQs')
    parser.add_argument('--rq', type=str, help='RQ to run (e.g. "2.1", "all")')
    parser.add_argument('--eval', action='store_true', help='Run only evaluation')
    parser.add_argument('--plot', action='store_true', help='Run only figure generation')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without executing')
    args = parser.parse_args()

    if args.list:
        list_rqs()
        return

    if not args.rq:
        parser.print_help()
        return

    if args.rq == 'all':
        for rq_id in RQ_REGISTRY:
            info = RQ_REGISTRY[rq_id]
            if info['evaluator'] or info['figure']:
                try:
                    run_rq(rq_id, eval_only=args.eval, plot_only=args.plot,
                           dry_run=args.dry_run)
                except NotImplementedError as e:
                    print(f"  [STUB] RQ {rq_id}: {e}")
        return

    if args.rq not in RQ_REGISTRY:
        print(f"Unknown RQ: {args.rq}")
        print(f"Available: {', '.join(RQ_REGISTRY.keys())}")
        return

    run_rq(args.rq, eval_only=args.eval, plot_only=args.plot,
           dry_run=args.dry_run)


if __name__ == '__main__':
    main()
