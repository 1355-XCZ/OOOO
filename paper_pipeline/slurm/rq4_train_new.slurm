#!/bin/bash
#SBATCH --job-name=rq4_train
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_train/%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_train/%a.err
#SBATCH --array=0-79

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

# 4 configs × 4 datasets × 5 tasks (1 balanced + 4 biased emotions) = 80
CONFIGS=(2x48 2x64 2x128 2x256)
LAYERS=(48 64 128 256)
DATASETS=(esd_en iemocap ravdess cremad)
EMOTIONS=(angry happy neutral sad)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_DATASETS=${#DATASETS[@]}       # 4
N_TASKS_PER_DS=5                  # 1 balanced + 4 biased
N_TASKS_PER_CFG=$((N_DATASETS * N_TASKS_PER_DS))  # 20

CFG_IDX=$((TASK_ID / N_TASKS_PER_CFG))
REMAINDER=$((TASK_ID % N_TASKS_PER_CFG))
DS_IDX=$((REMAINDER / N_TASKS_PER_DS))
SUBTASK=$((REMAINDER % N_TASKS_PER_DS))

CB_CONFIG=${CONFIGS[$CFG_IDX]}
NUM_LAYERS=${LAYERS[$CFG_IDX]}
DATASET=${DATASETS[$DS_IDX]}

echo "Task $TASK_ID: config=$CB_CONFIG layers=$NUM_LAYERS dataset=$DATASET subtask=$SUBTASK"

if [ "$SUBTASK" -eq 0 ]; then
    echo "Training BALANCED codebook: $CB_CONFIG / $DATASET"
    python BiasedCodebookExp_v2/scripts/train/train_balanced_codebook.py \
        --dataset "${DATASET}" \
        --ssl-model e2v \
        --codebook-config "${CB_CONFIG}" \
        --num-layers "${NUM_LAYERS}" \
        --codebook-size 2 \
        --device cuda
else
    EMO_IDX=$((SUBTASK - 1))
    EMOTION=${EMOTIONS[$EMO_IDX]}

    echo "Training BIASED codebook: $CB_CONFIG / $DATASET / $EMOTION"
    python BiasedCodebookExp_v2/scripts/train/train_biased_codebook.py \
        --dataset "${DATASET}" \
        --emotion "${EMOTION}" \
        --ssl-model e2v \
        --codebook-config "${CB_CONFIG}" \
        --num-layers "${NUM_LAYERS}" \
        --codebook-size 2 \
        --device cuda
fi
