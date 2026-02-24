#!/bin/bash
#SBATCH --job-name=rq32_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq32_eval_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq32_eval_%A_%a.err

# ==============================================================
# RQ3.2: Evaluate ambiguity codebooks (IEMOCAP source, OOD)
#
# 3 SSL × 3 OOD targets = 9 tasks
#
# Array mapping (0-8):
#   0-2:  e2v     2x24    (esd_en, ravdess, cremad)
#   3-5:  hubert  1024x24 (esd_en, ravdess, cremad)
#   6-8:  wavlm   1024x24 (esd_en, ravdess, cremad)
#
# Usage: sbatch --array=0-8 rq3_2_eval.slurm
# ==============================================================

source "$(dirname "$0")/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
OOD_TARGETS=(esd_en ravdess cremad)

SSL_IDX=$(( TASK_ID / 3 ))
TGT_IDX=$(( TASK_ID % 3 ))
TGT=${OOD_TARGETS[$TGT_IDX]}

case $SSL_IDX in
    0) SSL="e2v";    CB_CONFIG="2x24";    N_LAYERS=24 ;;
    1) SSL="hubert"; CB_CONFIG="1024x24"; N_LAYERS=24 ;;
    2) SSL="wavlm";  CB_CONFIG="1024x24"; N_LAYERS=24 ;;
esac

echo "Task $TASK_ID: ssl=$SSL cb=$CB_CONFIG test=$TGT"

python -m paper_pipeline.evaluators.rq3_2_ambiguity_table \
    --ssl-model "$SSL" \
    --codebook-config "$CB_CONFIG" \
    --test-dataset "$TGT" \
    --num-layers "$N_LAYERS" \
    --max-samples 200 \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
