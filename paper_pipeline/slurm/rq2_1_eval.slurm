#!/bin/bash
#SBATCH --job-name=rq21_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq21_eval_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq21_eval_%A_%a.err

# ==============================================================
# RQ2.1 Evaluation: SER-F1 for Matched/Unmatched/Balanced (e2v, 128x8)
#
# 4 datasets x 4 targets = 16 tasks (4 ID + 12 OOD)
#
# Array task mapping:
#   TASK_ID = SRC_IDX * 4 + TGT_IDX
#   SRC_IDX=TGT_IDX -> ID,  otherwise -> OOD
#
# Usage:
#   sbatch --array=0-15 rq2_1_eval.slurm
# ==============================================================

source "$(dirname "$0")/env.sh"

DATASETS=(esd_en iemocap ravdess cremad)

TASK_ID=${SLURM_ARRAY_TASK_ID}
SRC_IDX=$(( TASK_ID / 4 ))
TGT_IDX=$(( TASK_ID % 4 ))

SRC=${DATASETS[$SRC_IDX]}
TGT=${DATASETS[$TGT_IDX]}

if [ "$SRC" == "$TGT" ]; then
    echo "RQ2.1 Evaluating: $SRC (ID) | task=$TASK_ID"
    python -m paper_pipeline.evaluators.rq2_1_matched_ser \
        --codebook-dataset "$SRC" \
        --max-samples 200 \
        --force \
        --device cuda
else
    echo "RQ2.1 Evaluating: $SRC -> $TGT (OOD) | task=$TASK_ID"
    python -m paper_pipeline.evaluators.rq2_1_matched_ser \
        --codebook-dataset "$SRC" \
        --test-dataset "$TGT" \
        --max-samples 200 \
        --force \
        --device cuda
fi

echo "Done: task $TASK_ID at $(date)"
