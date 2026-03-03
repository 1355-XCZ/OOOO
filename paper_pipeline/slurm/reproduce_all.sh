#!/usr/bin/env bash
# ============================================================
# One-command Full Reproduction Pipeline
#
# Submits all SLURM jobs with correct dependency chains:
#   Step 0: Data preparation (CPU)
#   Step 1: Codebook & classifier training (GPU)
#   Step 2: Evaluation (GPU + CPU)
#   Step 3: Figure & table generation (CPU)
#
# Prerequisites:
#   - local_config.sh configured with correct paths
#   - All datasets available under DATA_ROOT
#   - python verify_env.py passes
#
# Usage:
#   bash paper_pipeline/slurm/reproduce_all.sh
#   bash paper_pipeline/slurm/reproduce_all.sh --dry-run   # print without submitting
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${EXP_ROOT}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Commands will be printed but not executed."
fi

# Load environment to ensure paths are set
if [[ -f "${EXP_ROOT}/local_config.sh" ]]; then
    source "${EXP_ROOT}/local_config.sh"
fi

SLURM_DIR="paper_pipeline/slurm"
LOG_DIR="${SLURM_DIR}/logs"
mkdir -p "${LOG_DIR}"

submit() {
    local desc="$1"; shift
    if [[ "${DRY_RUN}" == true ]]; then
        echo "  [DRY] $desc"
        echo "        sbatch $*"
        echo "DRY_${RANDOM}"
        return
    fi
    local out
    out=$(sbatch "$@" 2>&1)
    local job_id
    job_id=$(echo "${out}" | grep -oP '\d+$')
    echo "  [SUBMITTED] ${desc}  ->  Job ${job_id}"
    echo "${job_id}"
}

dep_flag() {
    # Build --dependency=afterok:id1:id2:... from arguments
    local ids=()
    for id in "$@"; do
        [[ "${id}" =~ ^DRY_ ]] && continue
        ids+=("${id}")
    done
    if [[ ${#ids[@]} -eq 0 ]]; then
        echo ""
    else
        local joined
        joined=$(IFS=:; echo "${ids[*]}")
        echo "--dependency=afterok:${joined}"
    fi
}

echo "============================================================"
echo "  BiasedCodebookExp_v2 -- Full Reproduction Pipeline"
echo "  EXP_ROOT: ${EXP_ROOT}"
echo "============================================================"

# ==============================================================
# Step 0: Data Preparation (runs on login node, fast)
# ==============================================================
echo ""
echo "--- Step 0: Data Preparation ---"

if [[ "${DRY_RUN}" == false ]]; then
    echo "  Running prepare_splits.py ..."
    python scripts/utils/prepare_splits.py
    echo "  Running prepare_ambiguity_splits.py ..."
    python scripts/utils/prepare_ambiguity_splits.py
    echo "  Running prepare_splits.py --cameo --test-only ..."
    python scripts/utils/prepare_splits.py --cameo --test-only
    echo "  Running prepare_splits.py --datasets msp --test-only ..."
    python scripts/utils/prepare_splits.py --datasets msp --test-only
    echo "  [DONE] Data preparation complete"
else
    echo "  [DRY] python scripts/utils/prepare_splits.py"
    echo "  [DRY] python scripts/utils/prepare_ambiguity_splits.py"
    echo "  [DRY] python scripts/utils/prepare_splits.py --cameo --test-only"
    echo "  [DRY] python scripts/utils/prepare_splits.py --datasets msp --test-only"
fi

# ==============================================================
# Step 1: Training (GPU)
# ==============================================================
echo ""
echo "--- Step 1: Codebook & Classifier Training ---"

JOB_RQ12_TRAIN=$(submit "RQ1+RQ2 codebooks (80 tasks)" \
    --array=0-79 "${SLURM_DIR}/train_multi_configs.slurm")

JOB_RQ3_TRAIN=$(submit "RQ3 mixed-ratio codebooks (64 tasks)" \
    --array=0-63 "${SLURM_DIR}/train_rq3_1_mixed.slurm")

JOB_RQ4_TRAIN=$(submit "RQ4 multi-config codebooks (288 tasks)" \
    --array=0-287 "${SLURM_DIR}/train_rq4_configs.slurm")

# SER classifiers need to be trained sequentially per (ssl, dataset) pair.
# Submit as a simple batch job that trains all 8 combinations.
SER_SCRIPT="${LOG_DIR}/_train_ser_all.sh"
cat > "${SER_SCRIPT}" << 'SEOF'
#!/bin/bash
#SBATCH --job-name=ser_train
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"
set -e

for SSL in hubert wavlm; do
  for DS in esd_en iemocap ravdess cremad; do
    echo "Training SER classifier: ${SSL} / ${DS}"
    python scripts/train/train_ser_classifier.py --ssl-model "${SSL}" --dataset "${DS}"
  done
done
echo "All SER classifiers trained."
SEOF

JOB_SER=$(submit "SER classifiers (8 combinations)" "${SER_SCRIPT}")

# ==============================================================
# Step 2: Evaluation (GPU + CPU)
# ==============================================================
echo ""
echo "--- Step 2: Evaluation ---"

DEP_RQ12=$(dep_flag "${JOB_RQ12_TRAIN}" "${JOB_SER}")
DEP_RQ3=$(dep_flag "${JOB_RQ12_TRAIN}" "${JOB_RQ3_TRAIN}" "${JOB_SER}")
DEP_RQ4=$(dep_flag "${JOB_RQ4_TRAIN}")

JOB_RQ1_EVAL=$(submit "RQ1 evaluation (36 tasks)" \
    ${DEP_RQ12:+${DEP_RQ12}} --array=0-35 "${SLURM_DIR}/rq1_eval.slurm")

JOB_RQ2_SER=$(submit "RQ2 SER recall (36 tasks)" \
    ${DEP_RQ12:+${DEP_RQ12}} --array=0-35 "${SLURM_DIR}/rq2_ser_eval.slurm")

JOB_RQ2_ENT=$(submit "RQ2 entropy (36 tasks)" \
    ${DEP_RQ12:+${DEP_RQ12}} --array=0-35 "${SLURM_DIR}/rq2_entropy_eval.slurm")

JOB_RQ2_CE=$(submit "RQ3 CE evaluation (9 tasks)" \
    ${DEP_RQ3:+${DEP_RQ3}} --array=0-8 "${SLURM_DIR}/rq2_ce_eval.slurm")

JOB_RQ4_EVAL=$(submit "RQ4 biased evaluation (32 tasks)" \
    ${DEP_RQ4:+${DEP_RQ4}} --array=0-31 "${SLURM_DIR}/rq4_eval.slurm")

DEP_RQ4_F1=$(dep_flag "${JOB_RQ4_EVAL}")
JOB_RQ4_F1=$(submit "RQ4 F1 post-processing (32 tasks, CPU)" \
    ${DEP_RQ4_F1:+${DEP_RQ4_F1}} --array=0-31 "${SLURM_DIR}/rq4_compute_f1.slurm")

JOB_RQ4_RATIO=$(submit "RQ4 ratio evaluation (32 tasks)" \
    ${DEP_RQ4:+${DEP_RQ4}} --array=0-31 "${SLURM_DIR}/rq4_ratio_eval.slurm")

DEP_RQ4_RATIO_F1=$(dep_flag "${JOB_RQ4_RATIO}")
JOB_RQ4_RATIO_F1=$(submit "RQ4 ratio F1 (32 tasks, CPU)" \
    ${DEP_RQ4_RATIO_F1:+${DEP_RQ4_RATIO_F1}} --array=0-31 "${SLURM_DIR}/rq4_ratio_f1.slurm")

# ==============================================================
# Step 3: Figure & Table Generation (CPU)
# ==============================================================
echo ""
echo "--- Step 3: Figure & Table Generation ---"

ALL_EVAL_JOBS="${JOB_RQ1_EVAL} ${JOB_RQ2_SER} ${JOB_RQ2_ENT} ${JOB_RQ2_CE} ${JOB_RQ4_F1} ${JOB_RQ4_RATIO_F1}"
DEP_FIGS=$(dep_flag ${ALL_EVAL_JOBS})

FIG_SCRIPT="${LOG_DIR}/_generate_figures.sh"
cat > "${FIG_SCRIPT}" << 'FEOF'
#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"
set -e

python -m paper_pipeline.pipeline --rq all --plot
echo "All figures and tables generated in results/paper_figures_rq/"
FEOF

JOB_FIGS=$(submit "Generate all figures & tables (CPU)" \
    ${DEP_FIGS:+${DEP_FIGS}} "${FIG_SCRIPT}")

# ==============================================================
# Summary
# ==============================================================
echo ""
echo "============================================================"
echo "  All jobs submitted. Dependency chain:"
echo ""
echo "  Training:    RQ1+2=${JOB_RQ12_TRAIN}  RQ3=${JOB_RQ3_TRAIN}  RQ4=${JOB_RQ4_TRAIN}  SER=${JOB_SER}"
echo "  Evaluation:  RQ1=${JOB_RQ1_EVAL}  RQ2-SER=${JOB_RQ2_SER}  RQ2-ENT=${JOB_RQ2_ENT}  RQ2-CE=${JOB_RQ2_CE}"
echo "               RQ4=${JOB_RQ4_EVAL}  RQ4-F1=${JOB_RQ4_F1}  RQ4-R=${JOB_RQ4_RATIO}  RQ4-RF1=${JOB_RQ4_RATIO_F1}"
echo "  Figures:     ${JOB_FIGS}"
echo ""
echo "  Monitor:  squeue -u \$(whoami)"
echo "  Logs:     ${LOG_DIR}/"
echo "============================================================"
