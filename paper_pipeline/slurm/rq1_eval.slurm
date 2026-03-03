#!/bin/bash
#SBATCH --job-name=rq1_eval
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq1_eval_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq1_eval_%A_%a.err

# ==============================================================
# RQ1: Evaluate BALANCED codebooks -- OOD cross-dataset evaluation
#
# 3 SSL configs × 4 sources × 3 OOD targets = 36 tasks
#
# Array mapping (0-35):
#   ssl_idx  = TASK_ID / 12
#   pair_idx = TASK_ID % 12  (12 OOD pairs: 4 sources × 3 targets)
#
# Usage: sbatch --array=0-35 rq1_eval.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"
set -e

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)

SSL_IDX=$(( TASK_ID / 12 ))
PAIR_IDX=$(( TASK_ID % 12 ))

case $SSL_IDX in
    0) SSL="e2v";    CB_CONFIG="2x24";    N_LAYERS=24; TAG="e2v_2x24"       ;;
    1) SSL="hubert"; CB_CONFIG="1024x24"; N_LAYERS=24; TAG="hubert_1024x24" ;;
    2) SSL="wavlm";  CB_CONFIG="1024x24"; N_LAYERS=24; TAG="wavlm_1024x24"  ;;
esac

# Build OOD pair from PAIR_IDX (0-11)
idx=0
for src in "${DATASETS[@]}"; do
    for tgt in "${DATASETS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            if [ $idx -eq $PAIR_IDX ]; then
                SRC="$src"
                TGT="$tgt"
            fi
            idx=$((idx+1))
        fi
    done
done

echo "Task $TASK_ID: ssl=$SSL config=$CB_CONFIG codebook_src=$SRC test=$TGT"

OUTPUT_DIR="${EXP_ROOT}/results/rq1_balanced/${TAG}"

python -m paper_pipeline.evaluators.rq1_evaluate \
    --ssl-model "$SSL" \
    --codebook-dataset "$SRC" \
    --test-dataset "$TGT" \
    --codebook-config "$CB_CONFIG" \
    --num-layers "$N_LAYERS" \
    --max-samples 200 \
    --metric cosine \
    --output-dir "$OUTPUT_DIR" \
    --emotions angry,happy,neutral,sad \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
