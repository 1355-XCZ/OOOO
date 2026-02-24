#!/bin/bash
# =================================================================
# RQ3.1 Extended -- Train new ratios + sample-level evaluation
#
# Two stages:
#   1. GPU training for 5 new ratios (240 array tasks)
#   2. GPU sample-level evaluation for ALL 10 ratios (36 array tasks)
#
# Usage:
#   bash submit_rq31_extended.sh               # full pipeline
#   bash submit_rq31_extended.sh --skip-train   # skip training
#   bash submit_rq31_extended.sh --dry-run
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_TRAIN=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=true ;;
        --dry-run)    DRY_RUN=true ;;
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

# Stage 1: Train new ratio codebooks
if $SKIP_TRAIN; then
    echo "=== Stage 1: SKIPPED (--skip-train) ==="
    TRAIN_JID=""
else
    echo "=== Stage 1: Train 5 new ratios (240 tasks) ==="
    TRAIN_JID=$(submit sbatch "${SCRIPT_DIR}/train_rq31_new_ratios.slurm")
    echo "  Submitted training job: ${TRAIN_JID}"
fi

# Stage 2: Sample-level evaluation
echo "=== Stage 2: Sample-level evaluation (36 tasks) ==="
DEP_ARG=""
if [ -n "${TRAIN_JID:-}" ] && [ "${TRAIN_JID}" != "DRY_JOB_ID" ]; then
    DEP_ARG="--dependency=afterok:${TRAIN_JID}"
fi
EVAL_JID=$(submit sbatch ${DEP_ARG} "${SCRIPT_DIR}/rq31_sample_eval.slurm")
echo "  Submitted eval job: ${EVAL_JID}"

echo ""
echo "=== Pipeline submitted ==="
[ -n "${TRAIN_JID:-}" ] && echo "  Train: ${TRAIN_JID}"
echo "  Eval:  ${EVAL_JID}"
