#!/bin/bash
# ==============================================================
# Submit RQ2.2: [Train 128x8 codebooks] -> Eval -> Table
#
# Usage:
#   bash paper_pipeline/slurm/submit_rq2_2.sh [--dry-run]
#   bash paper_pipeline/slurm/submit_rq2_2.sh --skip-train [--dry-run]
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
echo "  RQ2.2: SSL Comparison Table (128x8)"
echo "============================================"

TRAIN_DEP=""
if ! $SKIP_TRAIN; then
    echo ""
    echo "--- Train 128x8 codebooks for hubert + wavlm (40 tasks) ---"
    TRAIN_JID=$(submit "sbatch --array=0-39 --parsable ${SLURM_DIR}/train_128x8_hubert_wavlm.slurm")
    TRAIN_DEP="--dependency=afterok:${TRAIN_JID}"
    echo "  Train job: $TRAIN_JID"
else
    echo ""
    echo "--- Skipping training (--skip-train) ---"
fi

echo ""
echo "--- Evaluation (48 GPU tasks: 3 SSLs x 16 pairs) ---"
EVAL_JID=$(submit "sbatch --array=0-47 ${TRAIN_DEP} --parsable ${SLURM_DIR}/rq2_2_eval.slurm")

echo ""
echo "--- Table generation (CPU, after eval) ---"
TABLE_JID=$(submit "sbatch --dependency=afterok:${EVAL_JID} --parsable \
    --job-name=rq22_table \
    --partition=sapphire \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --output=paper_pipeline/slurm/logs/rq22_table_%j.out \
    --error=paper_pipeline/slurm/logs/rq22_table_%j.err \
    --wrap='source ${ENV_SH} && python -m paper_pipeline.pipeline --rq 2.2 --plot'")

echo ""
echo "============================================"
echo "  Jobs submitted!"
if ! $SKIP_TRAIN; then
    echo "  Train: ${TRAIN_JID} (40 array tasks)"
fi
echo "  Eval:  ${EVAL_JID} (48 array tasks)"
echo "  Table: ${TABLE_JID}"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Output:  results/paper_figures_rq/rq2_2_table.tex"
echo "============================================"
