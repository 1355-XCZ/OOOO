#!/bin/bash
#SBATCH --job-name=rq1_train
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq1_train_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq1_train_%A_%a.err

# ==============================================================
# Train 4 natural codebooks on Emilia EN for RQ1
#
# Array mapping (0-3):
#   0: e2v     2x24
#   1: e2v     4x24
#   2: hubert  1024x24
#   3: wavlm   1024x24
#
# Usage: sbatch --array=0-3 train_rq1_emilia.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}

case $TASK_ID in
    0) SSL="e2v";    CB_SIZE=2;    N_LAYERS=24 ;;
    1) SSL="e2v";    CB_SIZE=4;    N_LAYERS=24 ;;
    2) SSL="hubert"; CB_SIZE=1024; N_LAYERS=24 ;;
    3) SSL="wavlm";  CB_SIZE=1024; N_LAYERS=24 ;;
esac

echo "Task $TASK_ID: ssl=$SSL codebook_size=$CB_SIZE num_layers=$N_LAYERS"

python BiasedCodebookExp_v2/scripts/train/train_emilia_codebook.py \
    --ssl-model "$SSL" \
    --codebook-size "$CB_SIZE" \
    --num-layers "$N_LAYERS" \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
