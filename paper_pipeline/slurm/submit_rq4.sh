#!/bin/bash
# =================================================================
# RQ4 Master Submission Script
#
# Three stages:
#   1. GPU eval for ALL 23 configs (92 array tasks)
#   2. CPU F1 post-processing for all 23 configs (92 array tasks)
#   3. CPU table generation (1 task)
#
# Usage:
#   bash submit_rq4.sh              # full pipeline
#   bash submit_rq4.sh --skip-eval  # skip GPU eval (if sample_data already exists)
#   bash submit_rq4.sh --dry-run    # print commands without submitting
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_EVAL=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-eval) SKIP_EVAL=true ;;
        --dry-run)   DRY_RUN=true ;;
    esac
done

source "${SCRIPT_DIR}/env.sh"

submit() {
    if $DRY_RUN; then
        echo "[DRY RUN] $*"
        echo "DRY_JOB_ID"
    else
        local out
        out=$(eval "$@")
        local jid
        jid=$(echo "$out" | grep -oP '\d+')
        echo "$jid"
    fi
}

# Stage 1: GPU evaluation for ALL configs
if $SKIP_EVAL; then
    echo "=== Stage 1: SKIPPED (--skip-eval) ==="
    EVAL_JID=""
else
    echo "=== Stage 1: GPU evaluation (23 configs × 4 sources = 92 tasks) ==="
    EVAL_JID=$(submit sbatch "${SCRIPT_DIR}/rq4_eval.slurm")
    echo "  Submitted eval job: ${EVAL_JID}"
fi

# Stage 2: CPU F1 post-processing
echo "=== Stage 2: F1 post-processing (23 configs × 4 sources = 92 tasks) ==="
DEP_ARG=""
if [ -n "${EVAL_JID:-}" ] && [ "${EVAL_JID}" != "DRY_JOB_ID" ]; then
    DEP_ARG="--dependency=afterok:${EVAL_JID}"
fi
F1_JID=$(submit sbatch ${DEP_ARG} "${SCRIPT_DIR}/rq4_compute_f1.slurm")
echo "  Submitted F1 job: ${F1_JID}"

# Stage 3: Table generation
echo "=== Stage 3: Table generation ==="
DEP_ARG2=""
if [ -n "${F1_JID:-}" ] && [ "${F1_JID}" != "DRY_JOB_ID" ]; then
    DEP_ARG2="--dependency=afterok:${F1_JID}"
fi
TABLE_JID=$(submit sbatch ${DEP_ARG2} \
    --job-name=rq4_table \
    --partition=sapphire \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
    --time=00:10:00 \
    --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_table_%j.out \
    --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_table_%j.err \
    --wrap="'source ${SCRIPT_DIR}/env.sh && python -m paper_pipeline.figures.rq4'")
echo "  Submitted table job: ${TABLE_JID}"

echo ""
echo "=== Pipeline submitted ==="
[ -n "${EVAL_JID:-}" ] && echo "  Eval:  ${EVAL_JID}"
echo "  F1:    ${F1_JID}"
echo "  Table: ${TABLE_JID}"
