#!/bin/bash
# ==============================================================
# RQ3.2 Master Submission Script
#
# Stage 1: Train ambiguity codebooks (36 tasks)
# Stage 2: Evaluate (9 OOD tasks)
# Stage 3: Generate LaTeX table
#
# Usage:
#   bash submit_rq3_2.sh                  # full pipeline
#   bash submit_rq3_2.sh --skip-train     # skip training
#   bash submit_rq3_2.sh --dry-run        # just print commands
# ==============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$EXP_ROOT"

SKIP_TRAIN=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --dry-run)    DRY_RUN=true ;;
    esac
done

echo "========================================="
echo " RQ3.2: Ambiguity Codebook Table Pipeline"
echo "========================================="

# Stage 1: Train ambiguity codebooks
TRAIN_SLURM="$SCRIPT_DIR/train_rq32_ambiguity.slurm"

if [ "$SKIP_TRAIN" = true ]; then
    echo "[Stage 1] Skipping training (--skip-train)"
    TRAIN_JOB=""
else
    CMD="sbatch --array=0-35 $TRAIN_SLURM"
    echo "[Stage 1] $CMD"
    if [ "$DRY_RUN" = false ]; then
        TRAIN_JOB=$(eval "$CMD" | awk '{print $NF}')
        echo "  -> Job ID: $TRAIN_JOB"
    else
        TRAIN_JOB="DRY"
    fi
fi

# Stage 2: Evaluate (depends on training)
EVAL_SLURM="$SCRIPT_DIR/rq3_2_eval.slurm"

if [ -n "${TRAIN_JOB:-}" ] && [ "$TRAIN_JOB" != "DRY" ]; then
    DEP="--dependency=afterok:$TRAIN_JOB"
else
    DEP=""
fi

CMD="sbatch --array=0-8 $DEP $EVAL_SLURM"
echo "[Stage 2] $CMD"
if [ "$DRY_RUN" = false ]; then
    EVAL_JOB=$(eval "$CMD" | awk '{print $NF}')
    echo "  -> Job ID: $EVAL_JOB"
else
    EVAL_JOB="DRY"
fi

# Stage 3: Generate table (depends on evaluation)
if [ "$EVAL_JOB" != "DRY" ]; then
    DEP3="--dependency=afterok:$EVAL_JOB"
else
    DEP3=""
fi

TABLE_CMD="python -m paper_pipeline.figures.rq3_2"
WRAP_CMD="sbatch --job-name=rq32_tab --partition=sapphire --nodes=1 --ntasks=1 \
--cpus-per-task=2 --mem=8G --time=00:30:00 \
--output=paper_pipeline/slurm/logs/rq32_tab_%j.out \
--error=paper_pipeline/slurm/logs/rq32_tab_%j.err \
$DEP3 --wrap=\"source $SCRIPT_DIR/env.sh && $TABLE_CMD\""

echo "[Stage 3] $WRAP_CMD"
if [ "$DRY_RUN" = false ]; then
    TABLE_JOB=$(eval "$WRAP_CMD" | awk '{print $NF}')
    echo "  -> Job ID: $TABLE_JOB"
else
    TABLE_JOB="DRY"
fi

echo ""
echo "========================================="
echo " Pipeline submitted!"
echo "   Train:    ${TRAIN_JOB:-skipped}"
echo "   Evaluate: $EVAL_JOB"
echo "   Table:    $TABLE_JOB"
echo "========================================="
