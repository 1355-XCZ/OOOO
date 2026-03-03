#!/bin/bash
#SBATCH --job-name=rq4_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq4_eval_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq4_eval_%A_%a.err
#SBATCH --array=0-31

# 8 configs × 4 sources = 32 tasks
# Each task evaluates ALL codebook types and saves per-sample data + 9-method results.

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"
set -e

CONFIGS=(2x32 2x64 2x128 4x32 32x8 64x8 128x8 512x16)
SOURCES=(esd_en iemocap ravdess cremad)

N_SOURCES=${#SOURCES[@]}
CFG_IDX=$((SLURM_ARRAY_TASK_ID / N_SOURCES))
SRC_IDX=$((SLURM_ARRAY_TASK_ID % N_SOURCES))

CB_CONFIG="${CONFIGS[$CFG_IDX]}"
SOURCE="${SOURCES[$SRC_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: config=${CB_CONFIG}, source=${SOURCE}"

python -m paper_pipeline.evaluators.rq4_evaluate \
    --codebook-source "${SOURCE}" \
    --codebook-config "${CB_CONFIG}" \
    --phase all \
    --max-samples 500 \
    --metric cosine \
    --device cuda

echo "Done: ${CB_CONFIG} / ${SOURCE}"
