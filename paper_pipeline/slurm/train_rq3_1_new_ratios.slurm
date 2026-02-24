#!/bin/bash
#SBATCH --job-name=rq3_1_train
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=paper_pipeline/slurm/logs/rq3_1_train_%A_%a.out
#SBATCH --error=paper_pipeline/slurm/logs/rq3_1_train_%A_%a.err
#SBATCH --array=0-239

# 5 ratios × 4 emotions × 4 datasets × 3 SSLs = 240 tasks
#
# Ratios: 0.10, 0.20, 0.40, 0.50, 0.70
# Emotions: angry, happy, neutral, sad
# Datasets: esd_en, iemocap, ravdess, cremad
# SSLs: e2v (2x24), hubert (1024x24), wavlm (1024x24)

source "$(dirname "$0")/env.sh"
set -e

RATIOS=(0.10 0.20 0.40 0.50 0.70)
EMOTIONS=(angry happy neutral sad)
DATASETS=(esd_en iemocap ravdess cremad)
SSLS=(e2v hubert wavlm)

# Decode task ID: ratio(5) × emotion(4) × dataset(4) × ssl(3)
N_SSLS=${#SSLS[@]}
N_DATASETS=${#DATASETS[@]}
N_EMOTIONS=${#EMOTIONS[@]}

SSL_IDX=$((SLURM_ARRAY_TASK_ID % N_SSLS))
REMAINING=$((SLURM_ARRAY_TASK_ID / N_SSLS))
DS_IDX=$((REMAINING % N_DATASETS))
REMAINING=$((REMAINING / N_DATASETS))
EMO_IDX=$((REMAINING % N_EMOTIONS))
RATIO_IDX=$((REMAINING / N_EMOTIONS))

SSL="${SSLS[$SSL_IDX]}"
DATASET="${DATASETS[$DS_IDX]}"
EMOTION="${EMOTIONS[$EMO_IDX]}"
RATIO="${RATIOS[$RATIO_IDX]}"

# Determine codebook config
if [ "$SSL" = "e2v" ]; then
    CB_SIZE=2
    CB_CONFIG="2x24"
else
    CB_SIZE=1024
    CB_CONFIG="1024x24"
fi
N_LAYERS=24

echo "Task ${SLURM_ARRAY_TASK_ID}: SSL=${SSL}, dataset=${DATASET}, emotion=${EMOTION}, ratio=${RATIO}, config=${CB_CONFIG}"

# All datasets use full emotion names (angry, happy, neutral, sad)
DS_EMOTION="$EMOTION"

python scripts/train/train_mixed_codebook.py \
    --dataset "${DATASET}" \
    --emotion "${DS_EMOTION}" \
    --target-ratio "${RATIO}" \
    --ssl-model "${SSL}" \
    --codebook-config "${CB_CONFIG}" \
    --num-layers "${N_LAYERS}" \
    --codebook-size "${CB_SIZE}" \
    --device cuda

echo "Done: ${SSL}/${DATASET}/${EMOTION}/r${RATIO}"
