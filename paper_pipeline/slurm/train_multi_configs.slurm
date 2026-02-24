#!/bin/bash
#SBATCH --job-name=train_cb
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=paper_pipeline/slurm/logs/train_cb_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/train_cb_%A_%a.err

# ==============================================================
# Train codebooks for multiple SSL/config combinations
#
# 4 configs x 4 datasets x 5 codebooks = 80 tasks
#
# Array mapping (0-79):
#   0-19:  e2v    2x24
#   20-39: e2v    4x24
#   40-59: hubert 1024x24
#   60-79: wavlm  1024x24
#
# Per config (20 tasks):
#   Per dataset (5 tasks): balanced, biased_angry, biased_happy, biased_neutral, biased_sad
#   Datasets: esd_en(0-4), iemocap(5-9), ravdess(10-14), cremad(15-19)
#
# Usage: sbatch --array=0-79 train_multi_configs.slurm
# ==============================================================

source "$(dirname "$0")/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)
EMOTIONS=(angry happy neutral sad)

CONFIG_IDX=$(( TASK_ID / 20 ))
LOCAL=$(( TASK_ID % 20 ))
DS_IDX=$(( LOCAL / 5 ))
CB_IDX=$(( LOCAL % 5 ))

DS=${DATASETS[$DS_IDX]}

case $CONFIG_IDX in
    0) SSL="e2v";    CB_SIZE=2;    N_LAYERS=24; CB_TAG="2x24"    ;;
    1) SSL="e2v";    CB_SIZE=4;    N_LAYERS=24; CB_TAG="4x24"    ;;
    2) SSL="hubert"; CB_SIZE=1024; N_LAYERS=24; CB_TAG="1024x24" ;;
    3) SSL="wavlm";  CB_SIZE=1024; N_LAYERS=24; CB_TAG="1024x24" ;;
esac

echo "Task $TASK_ID: ssl=$SSL config=${CB_TAG} dataset=$DS cb_idx=$CB_IDX"

COMMON_ARGS="--dataset $DS --ssl-model $SSL --codebook-size $CB_SIZE --num-layers $N_LAYERS --samples-per-emotion 500"

if [ $CB_IDX -eq 0 ]; then
    echo "Training BALANCED codebook for $SSL / $DS (${CB_TAG})"
    python scripts/train/train_balanced_codebook.py \
        $COMMON_ARGS --codebook-name balanced
else
    EMO_IDX=$(( CB_IDX - 1 ))
    EMOTION=${EMOTIONS[$EMO_IDX]}
    echo "Training BIASED codebook for $SSL / $DS / $EMOTION (${CB_TAG})"
    python scripts/train/train_biased_codebook.py \
        $COMMON_ARGS --emotion "$EMOTION"
fi

echo "Done: task $TASK_ID at $(date)"
