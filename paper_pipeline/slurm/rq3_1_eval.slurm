#!/bin/bash
#SBATCH --job-name=rq31_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq31_eval_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq31_eval_%A_%a.err

# ==============================================================
# RQ3.1: Evaluate ratio codebooks (all 10 types, per-layer)
#   balanced, r10, r20, r40, r50, r70, r80, r95, r99, biased
#
# 3 SSL configs × 12 OOD pairs = 36 tasks
#
# Array mapping (0-35):
#   0-11:  e2v     2x24    (12 OOD pairs)
#   12-23: hubert  1024x24 (12 OOD pairs)
#   24-35: wavlm   1024x24 (12 OOD pairs)
#
# Usage: sbatch --array=0-35 rq3_1_eval.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"
set -e

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)

SSL_IDX=$(( TASK_ID / 12 ))
LOCAL=$(( TASK_ID % 12 ))
SRC_IDX=$(( LOCAL / 3 ))
# OOD targets: skip self -> 3 targets per source
declare -a TARGETS
k=0
for ((j=0; j<4; j++)); do
    if [ $j -ne $SRC_IDX ]; then
        TARGETS[$k]=${DATASETS[$j]}
        k=$((k+1))
    fi
done
TGT_IDX=$(( LOCAL % 3 ))

SRC=${DATASETS[$SRC_IDX]}
TGT=${TARGETS[$TGT_IDX]}

case $SSL_IDX in
    0) SSL="e2v";    CB_CONFIG="2x24";    N_LAYERS=24 ;;
    1) SSL="hubert"; CB_CONFIG="1024x24"; N_LAYERS=24 ;;
    2) SSL="wavlm";  CB_CONFIG="1024x24"; N_LAYERS=24 ;;
esac

echo "Task $TASK_ID: ssl=$SSL config=$CB_CONFIG src=$SRC tgt=$TGT"

python -m paper_pipeline.evaluators.rq3_1_ratio_table \
    --ssl-model "$SSL" \
    --codebook-config "$CB_CONFIG" \
    --codebook-dataset "$SRC" \
    --test-dataset "$TGT" \
    --num-layers "$N_LAYERS" \
    --max-samples 200 \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
