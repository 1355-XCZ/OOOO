#!/bin/bash
#SBATCH --job-name=rq23_ent
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq23_ent_%A_%a.out
#SBATCH --error=BiasedCodebookExp_v2/paper_pipeline/slurm/logs/rq23_ent_%A_%a.err

# ==============================================================
# RQ2.3 Entropy Evaluation (e2v, 128x8)
#
# 16 tasks: 4 ID + 12 OOD (4 datasets x 4 datasets)
#
# Array mapping (0-15):
#   0-3:   ID   (esd_en, iemocap, ravdess, cremad)
#   4-15:  OOD  (all cross-dataset pairs)
#
# Usage: sbatch --array=0-15 rq2_3_eval.slurm
# ==============================================================

source "${SLURM_SUBMIT_DIR}/BiasedCodebookExp_v2/paper_pipeline/slurm/env.sh"

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASETS=(esd_en iemocap ravdess cremad)

PAIRS=()
for src in "${DATASETS[@]}"; do
    PAIRS+=("${src}:${src}")
done
for src in "${DATASETS[@]}"; do
    for tgt in "${DATASETS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            PAIRS+=("${src}:${tgt}")
        fi
    done
done

IFS=':' read -r SRC TGT <<< "${PAIRS[$TASK_ID]}"

echo "Task $TASK_ID: codebook=$SRC test=$TGT"

python -m paper_pipeline.evaluators.rq2_3_entropy \
    --codebook-dataset "$SRC" \
    --test-dataset "$TGT" \
    --max-samples 200 \
    --device cuda

echo "Done: task $TASK_ID at $(date)"
