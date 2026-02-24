#!/bin/bash
# ==============================================================
# Submit RQ1: [Train Emilia codebooks] -> Eval -> Plot
#
# Usage:
#   bash paper_pipeline/slurm/submit_rq1.sh [--dry-run] [--skip-train]
# ==============================================================

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_SH="${SLURM_DIR}/env.sh"
SKIP_TRAIN=false
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
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
echo "  RQ1: Balanced Codebook (Emilia EN)"
echo "============================================"

TRAIN_DEP=""
if ! $SKIP_TRAIN; then
    echo ""
    echo "--- Train 4 Emilia codebooks ---"
    TRAIN_JID=$(submit "sbatch --array=0-3 --parsable ${SLURM_DIR}/train_rq1_emilia.slurm")
    TRAIN_DEP="--dependency=afterok:${TRAIN_JID}"
    echo "  Train job: $TRAIN_JID"
else
    echo ""
    echo "--- Skipping training (--skip-train) ---"
fi

echo ""
echo "--- Evaluation (16 GPU tasks: 4 configs x 4 test datasets) ---"
EVAL_JID=$(submit "sbatch --array=0-15 ${TRAIN_DEP} --parsable ${SLURM_DIR}/rq1_eval.slurm")

echo ""
echo "--- Plot (CPU, after eval) ---"
PLOT_JID=$(submit "sbatch --dependency=afterok:${EVAL_JID} --parsable \
    --job-name=rq1_plot \
    --partition=sapphire \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq1_plot_%j.out \
    --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq1_plot_%j.err \
    --wrap='source ${ENV_SH} && python -m paper_pipeline.pipeline --rq 1 --plot'")

echo ""
echo "============================================"
echo "  Jobs submitted!"
if ! $SKIP_TRAIN; then
    echo "  Train: ${TRAIN_JID} (4 tasks)"
fi
echo "  Eval:  ${EVAL_JID} (16 tasks)"
echo "  Plot:  ${PLOT_JID}"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Output:  BiasedCodebookExp_v2/results/paper_figures_rq/rq1_balanced_ssl_ood.png"
echo "============================================"
