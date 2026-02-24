#!/bin/bash
#SBATCH --job-name=rq32_amb
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq32_amb_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq32_amb_%A_%a.err

# ==============================================================
# RQ3.2: Train ambiguity codebooks (IEMOCAP only)
#
# 3 SSL × 4 emotions × 3 levels = 36 tasks
#
# Per SSL (12 tasks):
#   Per emotion (3 tasks): high, mid, low
#   Emotions: angry, happy, neutral, sad
#
# Array mapping (0-35):
#   0-11:  e2v     2x24
#   12-23: hubert  1024x24
#   24-35: wavlm   1024x24
#
# Usage: sbatch --array=0-35 train_rq32_ambiguity.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
EMOTIONS=(angry happy neutral sad)
LEVELS=(high mid low)

SSL_IDX=$(( TASK_ID / 12 ))
LOCAL=$(( TASK_ID % 12 ))
EMO_IDX=$(( LOCAL / 3 ))
LVL_IDX=$(( LOCAL % 3 ))

EMO=${EMOTIONS[$EMO_IDX]}
LVL=${LEVELS[$LVL_IDX]}

case $SSL_IDX in
    0) SSL="e2v";    CB_SIZE=2;    N_LAYERS=24 ;;
    1) SSL="hubert"; CB_SIZE=1024; N_LAYERS=24 ;;
    2) SSL="wavlm";  CB_SIZE=1024; N_LAYERS=24 ;;
esac

echo "Task $TASK_ID: ssl=$SSL emotion=$EMO level=$LVL"

python BiasedCodebookExp_v2/scripts/train/train_ambiguity_codebook.py \
    --ssl-model "$SSL" \
    --dataset iemocap \
    --emotion "$EMO" \
    --level "$LVL" \
    --codebook-size "$CB_SIZE" \
    --num-layers "$N_LAYERS" \
    --batch-size 16 \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
