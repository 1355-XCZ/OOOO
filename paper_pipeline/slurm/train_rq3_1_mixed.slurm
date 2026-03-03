#!/bin/bash
#SBATCH --job-name=rq3_1_mix
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq3_1_mix_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq3_1_mix_%A_%a.err

# ==============================================================
# RQ3: Train mixed-ratio codebooks for 2 SSL models
#
# 2 SSL × 4 datasets × 4 emotions × 2 ratios = 64 tasks
#
# Per SSL (32 tasks):
#   Per dataset (8 tasks): 4 emotions × 2 ratios
#   Datasets: esd_en, iemocap, ravdess, cremad
#   Emotions: angry, happy, neutral, sad
#   Ratios:   0.95, 0.99
#
# Array mapping (0-63):
#   0-31:  e2v     2x24
#   32-63: hubert  1024x24
#
# Usage: sbatch --array=0-63 train_rq3_1_mixed.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)
EMOTIONS=(angry happy neutral sad)
RATIOS=(0.95 0.99)

SSL_IDX=$(( TASK_ID / 32 ))
LOCAL=$(( TASK_ID % 32 ))
DS_IDX=$(( LOCAL / 8 ))
EMO_IDX=$(( (LOCAL % 8) / 2 ))
RATIO_IDX=$(( LOCAL % 2 ))

DS=${DATASETS[$DS_IDX]}
EMO=${EMOTIONS[$EMO_IDX]}
RATIO=${RATIOS[$RATIO_IDX]}

case $SSL_IDX in
    0) SSL="e2v";    CB_SIZE=2;    N_LAYERS=24 ;;
    1) SSL="hubert"; CB_SIZE=1024; N_LAYERS=24 ;;
esac

echo "Task $TASK_ID: ssl=$SSL dataset=$DS emotion=$EMO ratio=$RATIO"

python scripts/train/train_mixed_codebook.py \
    --ssl-model "$SSL" \
    --dataset "$DS" \
    --emotion "$EMO" \
    --target-ratio "$RATIO" \
    --codebook-size "$CB_SIZE" \
    --num-layers "$N_LAYERS" \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
