#!/bin/bash
#SBATCH --job-name=rq2_ent
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq2_ent_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq2_ent_%A_%a.err

# ==============================================================
# RQ2 Entropy: Token Entropy across 3 SSL configs
#
# 3 SSL configs x 12 OOD pairs = 36 tasks
#
# Array mapping (0-35):
#   0-11:  e2v     2x24    (12 OOD pairs)
#   12-23: hubert  1024x24 (12 OOD pairs)
#   24-35: wavlm   1024x24 (12 OOD pairs)
#
# Usage: sbatch --array=0-35 rq2_entropy_eval.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)

# Build OOD pairs (src != tgt)
OOD_PAIRS=()
for src in "${DATASETS[@]}"; do
    for tgt in "${DATASETS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            OOD_PAIRS+=("${src}:${tgt}")
        fi
    done
done

CONFIG_IDX=$(( TASK_ID / 12 ))
PAIR_IDX=$(( TASK_ID % 12 ))

case $CONFIG_IDX in
    0) SSL="e2v";    CB_CONFIG="2x24";    N_LAYERS=24 ;;
    1) SSL="hubert"; CB_CONFIG="1024x24"; N_LAYERS=24 ;;
    2) SSL="wavlm";  CB_CONFIG="1024x24"; N_LAYERS=24 ;;
esac

IFS=':' read -r SRC TGT <<< "${OOD_PAIRS[$PAIR_IDX]}"

echo "Task $TASK_ID: ssl=$SSL config=$CB_CONFIG src=$SRC tgt=$TGT"

python -m paper_pipeline.evaluators.rq2_3_entropy \
    --ssl-model "$SSL" \
    --codebook-config "$CB_CONFIG" \
    --codebook-dataset "$SRC" \
    --test-dataset "$TGT" \
    --num-layers "$N_LAYERS" \
    --max-samples 200 \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
