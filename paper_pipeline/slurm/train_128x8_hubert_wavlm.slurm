#!/bin/bash
#SBATCH --job-name=train128x8
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=paper_pipeline/slurm/logs/train128x8_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/train128x8_%A_%a.err

# ==============================================================
# Train 128x8 codebooks for hubert & wavlm on 4 ID datasets
#
# 2 SSL models x 4 datasets x 5 codebooks = 40 tasks
#
# Array mapping (0-39):
#   0-19:  hubert (0-4 esd_en, 5-9 iemocap, 10-14 ravdess, 15-19 cremad)
#   20-39: wavlm  (same layout)
#
# Per dataset (5 tasks): balanced, biased_angry, biased_happy, biased_neutral, biased_sad
#
# Usage: sbatch --array=0-39 train_128x8_hubert_wavlm.slurm
# ==============================================================

source "$(dirname "$0")/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)
EMOTIONS=(angry happy neutral sad)

SSL_IDX=$(( TASK_ID / 20 ))
LOCAL=$(( TASK_ID % 20 ))
DS_IDX=$(( LOCAL / 5 ))
CB_IDX=$(( LOCAL % 5 ))

if [ $SSL_IDX -eq 0 ]; then SSL="hubert"; else SSL="wavlm"; fi
DS=${DATASETS[$DS_IDX]}

echo "Task $TASK_ID: ssl=$SSL dataset=$DS cb_idx=$CB_IDX config=128x8"

if [ $CB_IDX -eq 0 ]; then
    echo "Training BALANCED codebook for $SSL / $DS (128x8)"
    python scripts/train/train_balanced_codebook.py \
        --dataset "$DS" \
        --ssl-model "$SSL" \
        --codebook-size 128 \
        --num-layers 8 \
        --samples-per-emotion 500 \
        --codebook-name balanced
else
    EMO_IDX=$(( CB_IDX - 1 ))
    EMOTION=${EMOTIONS[$EMO_IDX]}
    echo "Training BIASED codebook for $SSL / $DS / $EMOTION (128x8)"
    python scripts/train/train_biased_codebook.py \
        --dataset "$DS" \
        --emotion "$EMOTION" \
        --ssl-model "$SSL" \
        --codebook-size 128 \
        --num-layers 8
fi

echo "Done: task $TASK_ID at $(date)"
