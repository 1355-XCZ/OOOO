#!/bin/bash
#SBATCH --job-name=rq2_ce
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq2_ce_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq2_ce_%A_%a.err

# ==============================================================
# RQ2 Cross-Entropy Distribution Preservation Evaluation
#
# 3 SSL models × 3 codebook sources = 9 tasks
#
# Array mapping (0-8):
#   0: e2v    esd_en     3: hubert esd_en     6: wavlm esd_en
#   1: e2v    ravdess    4: hubert ravdess    7: wavlm ravdess
#   2: e2v    cremad     5: hubert cremad     8: wavlm cremad
#
# Usage: sbatch --array=0-8 rq2_ce_eval.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"
set -e

SSL_MODELS=(e2v hubert wavlm)
SOURCES=(esd_en ravdess cremad)

N_SOURCES=${#SOURCES[@]}
SSL_IDX=$((SLURM_ARRAY_TASK_ID / N_SOURCES))
SRC_IDX=$((SLURM_ARRAY_TASK_ID % N_SOURCES))

SSL_MODEL="${SSL_MODELS[$SSL_IDX]}"
SOURCE="${SOURCES[$SRC_IDX]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ssl_model=${SSL_MODEL}, source=${SOURCE}"

python -m paper_pipeline.evaluators.rq2_ce \
    --ssl-model "${SSL_MODEL}" \
    --codebook-source "${SOURCE}" \
    --version va \
    --device cuda \
    --save-samples \
    --include-ratio

echo "Done: ${SSL_MODEL} / ${SOURCE} at $(date)"
