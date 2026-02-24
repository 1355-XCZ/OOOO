#!/bin/bash
# ==============================================================
# Submit RQ2.3: Eval (GPU) -> Plot (CPU)
#
# Usage:
#   bash paper_pipeline/slurm/submit_rq2_3.sh [--dry-run]
# ==============================================================

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_SH="${SLURM_DIR}/env.sh"
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
    esac
done

submit() {
    local cmd="$1"
    if [ -n "$DRY_RUN" ]; then
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
echo "  RQ2.3: Codebook Token Entropy (128x8)"
echo "============================================"

echo ""
echo "--- Evaluation (16 GPU tasks: 4 ID + 12 OOD) ---"
EVAL_JID=$(submit "sbatch --array=0-15 --parsable ${SLURM_DIR}/rq2_3_eval.slurm")

echo ""
echo "--- Plot (CPU, after eval) ---"
PLOT_JID=$(submit "sbatch --dependency=afterok:${EVAL_JID} --parsable \
    --job-name=rq23_plot \
    --partition=sapphire \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --output=paper_pipeline/slurm/logs/rq23_plot_%j.out \
    --error=paper_pipeline/slurm/logs/rq23_plot_%j.err \
    --wrap='source ${ENV_SH} && python -m paper_pipeline.pipeline --rq 2.3 --plot'")

echo ""
echo "============================================"
echo "  Jobs submitted!"
echo "  Eval: ${EVAL_JID} (16 array tasks)"
echo "  Plot: ${PLOT_JID}"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Output:  results/paper_figures_rq/rq2_3_entropy_ood.png"
echo "============================================"
