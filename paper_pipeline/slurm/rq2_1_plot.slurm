#!/bin/bash
#SBATCH --job-name=rq21_plot
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=paper_pipeline/slurm/logs/rq21_plot_%j.out
#SBATCH --error=paper_pipeline/slurm/logs/rq21_plot_%j.err

# ==============================================================
# RQ2.1 Plot: Generate SER-F1 figure (CPU only)
#
# Usage:
#   sbatch rq2_1_plot.slurm
#   sbatch --dependency=afterok:EVAL_JOB_ID rq2_1_plot.slurm
# ==============================================================

source "$(dirname "$0")/env.sh"

echo "Generating RQ2.1 figure..."
python -m paper_pipeline.pipeline --rq 2.1 --plot

echo "Done at $(date)"
echo "Figure: ${EXP_ROOT}/results/paper_figures/rq2_1_ser_f1_ood.png"
