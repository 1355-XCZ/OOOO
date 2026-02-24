#!/bin/bash
# ==============================================================
# Submit RQ2.1: Evaluation (GPU array) -> Plot (CPU, depends on eval)
#
# Usage:
#   bash paper_pipeline/slurm/submit_rq2_1.sh [--dry-run]
# ==============================================================

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN="$1"

submit() {
    local cmd="$1"
    if [ "$DRY_RUN" == "--dry-run" ]; then
        echo "  [DRY RUN] $cmd" >&2
        echo "DRYRUN_$$"
    else
        echo "  Submitting: $cmd" >&2
        local output
        output=$(eval "$cmd")
        local job_id=$(echo "$output" | grep -oP '\d+' | head -1)
        echo "    -> Job ID: $job_id" >&2
        echo "$job_id"
    fi
}

echo "============================================"
echo "  RQ2.1: Submit Evaluation + Plot"
echo "============================================"

echo ""
echo "--- Evaluation (16 GPU tasks) ---"
EVAL_JID=$(submit "sbatch --array=0-15 --parsable ${SLURM_DIR}/rq2_1_eval.slurm")

echo ""
echo "--- Plot (CPU, after eval completes) ---"
PLOT_JID=$(submit "sbatch --dependency=afterok:${EVAL_JID} --parsable ${SLURM_DIR}/rq2_1_plot.slurm")

echo ""
echo "============================================"
echo "  Jobs submitted!"
echo "  Eval:  ${EVAL_JID} (16 array tasks)"
echo "  Plot:  ${PLOT_JID} (depends on eval)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Figure:  BiasedCodebookExp_v2/results/paper_figures/rq2_1_ser_f1_ood.png"
echo "============================================"
