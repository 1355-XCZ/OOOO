#!/bin/bash
#SBATCH --job-name=rq3_1_samp
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq3_1_samp_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq3_1_samp_%A_%a.err
#SBATCH --array=0-35

# 3 SSLs × 12 OOD pairs = 36 tasks
# Evaluates ALL 10 ratio types with sample-level output

source "$(dirname "$0")/env.sh"

SSLS=(e2v hubert wavlm)
CB_CONFIGS=(2x24 1024x24 1024x24)
DATASETS=(esd_en iemocap ravdess cremad)

# Build OOD pairs
declare -a SRC_LIST TGT_LIST
idx=0
for src in "${DATASETS[@]}"; do
    for tgt in "${DATASETS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            SRC_LIST[$idx]="$src"
            TGT_LIST[$idx]="$tgt"
            idx=$((idx+1))
        fi
    done
done
N_PAIRS=$idx  # 12

SSL_IDX=$((SLURM_ARRAY_TASK_ID / N_PAIRS))
PAIR_IDX=$((SLURM_ARRAY_TASK_ID % N_PAIRS))

SSL="${SSLS[$SSL_IDX]}"
CB_CONFIG="${CB_CONFIGS[$SSL_IDX]}"
SRC="${SRC_LIST[$PAIR_IDX]}"
TGT="${TGT_LIST[$PAIR_IDX]}"
N_LAYERS=24

echo "Task ${SLURM_ARRAY_TASK_ID}: SSL=${SSL}, config=${CB_CONFIG}, ${SRC} -> ${TGT}"

python -m paper_pipeline.evaluators.rq3_1_sample_level \
    --ssl-model "${SSL}" \
    --codebook-config "${CB_CONFIG}" \
    --codebook-dataset "${SRC}" \
    --test-dataset "${TGT}" \
    --num-layers "${N_LAYERS}" \
    --max-samples 200 \
    --device cuda

echo "Done: ${SSL} ${SRC} -> ${TGT}"
