#!/bin/bash
#SBATCH --job-name=rq4_evnew
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_eval_new/%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_eval_new/%a.err
#SBATCH --array=0-15

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

# 4 configs × 4 sources = 16 tasks
CONFIGS=(2x48 2x64 2x128 2x256)
SOURCES=(esd_en iemocap ravdess cremad)

TASK_ID=${SLURM_ARRAY_TASK_ID}
CFG_IDX=$((TASK_ID / 4))
SRC_IDX=$((TASK_ID % 4))

CB_CONFIG=${CONFIGS[$CFG_IDX]}
SOURCE=${SOURCES[$SRC_IDX]}

echo "Task $TASK_ID: config=$CB_CONFIG source=$SOURCE"

python BiasedCodebookExp_v2/scripts/evaluate/evaluate_unified.py \
    --codebook-source "${SOURCE}" \
    --codebook-config "${CB_CONFIG}" \
    --phase all \
    --max-samples 500 \
    --metric cosine \
    --device cuda
