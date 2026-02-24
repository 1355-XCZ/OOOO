#!/bin/bash
# ==============================================================
# Submit RQ3.1: Train mixed codebooks -> Eval -> Table
#
# Assumes balanced + biased codebooks already trained (via train_multi_configs).
# This script trains mixed-ratio codebooks, then evaluates all 5 ratios,
# then generates the LaTeX table.
#
# Usage:
#   bash paper_pipeline/slurm/submit_rq3_1.sh [--skip-train] [--dry-run]
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
echo "  RQ3.1: Ratio Codebook Table"
echo "============================================"

TRAIN_DEP=""
if ! $SKIP_TRAIN; then
    echo ""
    echo "--- Train mixed-ratio codebooks (144 GPU tasks) ---"
    TRAIN_JID=$(submit "sbatch --array=0-143 --parsable ${SLURM_DIR}/train_rq31_mixed.slurm")
    TRAIN_DEP="--dependency=afterok:${TRAIN_JID}"
    echo "  Train job: $TRAIN_JID"
else
    echo ""
    echo "--- Skipping training (--skip-train) ---"
fi

echo ""
echo "--- Evaluation (36 GPU tasks: 3 SSL x 12 OOD pairs) ---"
EVAL_JID=$(submit "sbatch --array=0-35 ${TRAIN_DEP} --parsable ${SLURM_DIR}/rq3_1_eval.slurm")

echo ""
echo "--- Table generation (CPU, after eval) ---"
TABLE_JID=$(submit "sbatch --dependency=afterok:${EVAL_JID} --parsable \
    --job-name=rq31_tab \
    --partition=sapphire \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq31_table_%j.out \
    --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq31_table_%j.err \
    --wrap='source ${ENV_SH} && python -m paper_pipeline.figures.rq3_1'")

echo ""
echo "============================================"
echo "  Jobs submitted!"
if ! $SKIP_TRAIN; then
    echo "  Train: ${TRAIN_JID} (144 tasks)"
fi
echo "  Eval:  ${EVAL_JID} (36 tasks)"
echo "  Table: ${TABLE_JID}"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Output:  BiasedCodebookExp_v2/results/paper_figures_rq/rq3_1_table.tex"
echo "============================================"
