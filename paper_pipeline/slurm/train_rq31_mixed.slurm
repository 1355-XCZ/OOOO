#!/bin/bash
#SBATCH --job-name=rq31_mix
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq31_mix_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq31_mix_%A_%a.err

# ==============================================================
# RQ3.1: Train mixed-ratio codebooks for 3 SSL models
#
# 3 SSL × 4 datasets × 4 emotions × 3 ratios = 144 tasks
#
# Per SSL (48 tasks):
#   Per dataset (12 tasks): 4 emotions × 3 ratios
#   Datasets: esd_en, iemocap, ravdess, cremad
#   Emotions: angry, happy, neutral, sad
#   Ratios:   0.80, 0.95, 0.99
#
# Array mapping (0-143):
#   0-47:   e2v     2x24
#   48-95:  hubert  1024x24
#   96-143: wavlm   1024x24
#
# Usage: sbatch --array=0-143 train_rq31_mixed.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)
EMOTIONS=(angry happy neutral sad)
RATIOS=(0.80 0.95 0.99)

SSL_IDX=$(( TASK_ID / 48 ))
LOCAL=$(( TASK_ID % 48 ))
DS_IDX=$(( LOCAL / 12 ))
EMO_IDX=$(( (LOCAL % 12) / 3 ))
RATIO_IDX=$(( LOCAL % 3 ))

DS=${DATASETS[$DS_IDX]}
EMO=${EMOTIONS[$EMO_IDX]}
RATIO=${RATIOS[$RATIO_IDX]}

case $SSL_IDX in
    0) SSL="e2v";    CB_SIZE=2;    N_LAYERS=24 ;;
    1) SSL="hubert"; CB_SIZE=1024; N_LAYERS=24 ;;
    2) SSL="wavlm";  CB_SIZE=1024; N_LAYERS=24 ;;
esac

echo "Task $TASK_ID: ssl=$SSL dataset=$DS emotion=$EMO ratio=$RATIO"

python BiasedCodebookExp_v2/scripts/train/train_mixed_codebook.py \
    --ssl-model "$SSL" \
    --dataset "$DS" \
    --emotion "$EMO" \
    --target-ratio "$RATIO" \
    --codebook-size "$CB_SIZE" \
    --num-layers "$N_LAYERS" \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
