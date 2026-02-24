#!/bin/bash
#SBATCH --job-name=rq4_f1
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_f1_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq4_f1_%A_%a.err
#SBATCH --array=0-91

# 23 configs × 4 sources = 92 tasks (CPU only)

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"
set -e

CONFIGS=(2x16 2x24 2x32 2x48 2x64 2x128 2x256 4x16 4x32 8x24 16x16 16x24 16x32 32x8 64x8 128x8 128x12 128x16 128x24 256x12 512x16 1024x16 2048x16)
SOURCES=(esd_en iemocap ravdess cremad)

N_SOURCES=${#SOURCES[@]}
CFG_IDX=$((SLURM_ARRAY_TASK_ID / N_SOURCES))
SRC_IDX=$((SLURM_ARRAY_TASK_ID % N_SOURCES))

CB_CONFIG="${CONFIGS[$CFG_IDX]}"
SOURCE="${SOURCES[$SRC_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: config=${CB_CONFIG}, source=${SOURCE}"

python -m paper_pipeline.evaluators.rq4_compute_f1 \
    --codebook-config "${CB_CONFIG}" \
    --codebook-source "${SOURCE}"

echo "Done: ${CB_CONFIG} / ${SOURCE}"
