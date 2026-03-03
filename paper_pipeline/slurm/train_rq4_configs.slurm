#!/bin/bash
#SBATCH --job-name=rq4_train
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq4_train_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq4_train_%A_%a.err

# ==============================================================
# RQ4: Train codebooks for 8 e2v configs × 4 datasets
#
# Per config × dataset (9 codebooks):
#   0: balanced
#   1-4: biased (angry, happy, neutral, sad)
#   5-8: mixed_r99 (angry, happy, neutral, sad)
#
# 8 configs × 4 datasets × 9 codebooks = 288 tasks
#
# Array mapping (0-287):
#   config_idx = TASK_ID / 36
#   local = TASK_ID % 36
#   ds_idx = local / 9
#   cb_idx = local % 9
#
# Usage: sbatch --array=0-287 train_rq4_configs.slurm
#        sbatch --array=0-287%16 train_rq4_configs.slurm  # max 16 concurrent
# ==============================================================

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)
EMOTIONS=(angry happy neutral sad)

CONFIGS_CB_SIZE=(2 2 2 4 32 64 128 512)
CONFIGS_N_LAYERS=(32 64 128 32 8 8 8 16)
CONFIGS_TAG=(2x32 2x64 2x128 4x32 32x8 64x8 128x8 512x16)

CONFIG_IDX=$(( TASK_ID / 36 ))
LOCAL=$(( TASK_ID % 36 ))
DS_IDX=$(( LOCAL / 9 ))
CB_IDX=$(( LOCAL % 9 ))

DS=${DATASETS[$DS_IDX]}
CB_SIZE=${CONFIGS_CB_SIZE[$CONFIG_IDX]}
N_LAYERS=${CONFIGS_N_LAYERS[$CONFIG_IDX]}
CB_TAG=${CONFIGS_TAG[$CONFIG_IDX]}

COMMON_ARGS="--dataset $DS --ssl-model e2v --codebook-size $CB_SIZE --num-layers $N_LAYERS"

if [ $CB_IDX -eq 0 ]; then
    echo "Task $TASK_ID: BALANCED e2v / $DS (${CB_TAG})"
    python scripts/train/train_balanced_codebook.py \
        $COMMON_ARGS --codebook-name balanced
elif [ $CB_IDX -le 4 ]; then
    EMO_IDX=$(( CB_IDX - 1 ))
    EMOTION=${EMOTIONS[$EMO_IDX]}
    echo "Task $TASK_ID: BIASED e2v / $DS / $EMOTION (${CB_TAG})"
    python scripts/train/train_biased_codebook.py \
        $COMMON_ARGS --emotion "$EMOTION"
else
    EMO_IDX=$(( CB_IDX - 5 ))
    EMOTION=${EMOTIONS[$EMO_IDX]}
    echo "Task $TASK_ID: MIXED_R99 e2v / $DS / $EMOTION (${CB_TAG})"
    python scripts/train/train_mixed_codebook.py \
        $COMMON_ARGS --emotion "$EMOTION" --target-ratio 0.99
fi

echo "Done: task $TASK_ID at $(date)"
