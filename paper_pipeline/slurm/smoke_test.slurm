#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=paper_pipeline/slurm/logs/smoke_test_%j.out
#SBATCH --error=paper_pipeline/slurm/logs/smoke_test_%j.err

# ==============================================================
# Smoke Test on SLURM CPU node
#
# Usage: sbatch paper_pipeline/slurm/smoke_test.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"

bash "${EXP_ROOT}/run_smoke_test.sh"
