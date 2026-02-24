#!/bin/bash
#SBATCH --job-name=rq4_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_eval_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_eval_%A_%a.err
#SBATCH --array=0-43

# 11 missing configs × 4 sources = 44 tasks
# Runs evaluate_unified.py --phase all for each (config, source) pair

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

CONFIGS=(2x16 2x24 4x16 16x24 16x32 128x12 128x16 128x24 512x16 1024x16 2048x16)
SOURCES=(esd_en iemocap ravdess cremad)

N_SOURCES=${#SOURCES[@]}
CFG_IDX=$((SLURM_ARRAY_TASK_ID / N_SOURCES))
SRC_IDX=$((SLURM_ARRAY_TASK_ID % N_SOURCES))

CB_CONFIG="${CONFIGS[$CFG_IDX]}"
SOURCE="${SOURCES[$SRC_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: config=${CB_CONFIG}, source=${SOURCE}"

python BiasedCodebookExp_v2/scripts/evaluate/evaluate_unified.py \
    --codebook-source "${SOURCE}" \
    --codebook-config "${CB_CONFIG}" \
    --phase all \
    --max-samples 500 \
    --metric cosine \
    --device cuda

echo "Done: ${CB_CONFIG} / ${SOURCE}"
