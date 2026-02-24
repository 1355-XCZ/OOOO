#!/bin/bash
#SBATCH --job-name=rq22_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq22_eval_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq22_eval_%A_%a.err

# ==============================================================
# RQ2.2 Evaluation: L2 + SER-F1 for 3 SSL models (128x8)
#
# 3 SSLs x 16 pairs = 48 tasks
#
# Array mapping:
#   TASK_ID = SSL_IDX * 16 + PAIR_IDX
#   PAIR_IDX = SRC_IDX * 4 + TGT_IDX
#
# Usage: sbatch --array=0-47 rq2_2_eval.slurm
# ==============================================================

source "$(dirname "$0")/env.sh"

SSLS=(e2v hubert wavlm)
DATASETS=(esd_en iemocap ravdess cremad)

TASK_ID=${SLURM_ARRAY_TASK_ID}
SSL_IDX=$(( TASK_ID / 16 ))
PAIR_IDX=$(( TASK_ID % 16 ))
SRC_IDX=$(( PAIR_IDX / 4 ))
TGT_IDX=$(( PAIR_IDX % 4 ))

SSL=${SSLS[$SSL_IDX]}
SRC=${DATASETS[$SRC_IDX]}
TGT=${DATASETS[$TGT_IDX]}

if [ "$SRC" == "$TGT" ]; then
    echo "RQ2.2: $SSL / $SRC (ID) | task=$TASK_ID"
    python -m paper_pipeline.evaluators.rq2_2_ssl_table \
        --ssl-model "$SSL" \
        --codebook-dataset "$SRC" \
        --max-samples 200 \
        --device cuda
else
    echo "RQ2.2: $SSL / $SRC -> $TGT (OOD) | task=$TASK_ID"
    python -m paper_pipeline.evaluators.rq2_2_ssl_table \
        --ssl-model "$SSL" \
        --codebook-dataset "$SRC" \
        --test-dataset "$TGT" \
        --max-samples 200 \
        --device cuda
fi

echo "Done: task $TASK_ID at $(date)"
