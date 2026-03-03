#!/bin/bash
#SBATCH --job-name=rq4_ratio_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq4_ratio_eval_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq4_ratio_eval_%A_%a.err

# ==============================================================
# RQ4 Ratio Comprehensive Evaluation
#
# 8 configs × 4 sources = 32 tasks
# Each task evaluates biased + mixed_r99 codebook types
# and saves per-sample data + 9-method results.
#
# Usage:
#   sbatch --array=0-31 rq4_ratio_eval.slurm
#   sbatch --array=0-31%8 rq4_ratio_eval.slurm   # max 8 concurrent
# ==============================================================

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

python -m paper_pipeline.evaluators.rq4_ratio_evaluate \
    --codebook-source "${SOURCE}" \
    --codebook-config "${CB_CONFIG}" \
    --max-samples 500 \
    --device cuda \
    --force

echo "Done: ${CB_CONFIG} / ${SOURCE} at $(date)"
