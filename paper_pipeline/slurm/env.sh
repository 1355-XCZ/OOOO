#!/bin/bash
# ============================================================
# Paper Pipeline -- Shared SLURM Environment
#
# Source this at the top of any paper_pipeline .slurm script:
#   source "${SLURM_SUBMIT_DIR}/paper_pipeline/slurm/env.sh"
#
# All paths are auto-detected from this script's location.
# User-specific overrides go in local_config.sh (see template).
# ============================================================

# --- Auto-detect project layout from this script's location ---
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR}" && ! -f "${_SCRIPT_DIR}/env.sh" ]]; then
    _SCRIPT_DIR="${SLURM_SUBMIT_DIR}/paper_pipeline/slurm"
fi
export EXP_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

# --- Load user-specific overrides (venv path, data root, etc.) ---
_LOCAL_CFG="${EXP_ROOT}/local_config.sh"
if [[ -f "${_LOCAL_CFG}" ]]; then
    source "${_LOCAL_CFG}"
fi

# --- Defaults (override via local_config.sh or environment) ---
export VENV_PATH="${VENV_PATH:-${EXP_ROOT}/.venv}"
export DATA_ROOT="${DATA_ROOT:-${EXP_ROOT}/download}"
export E2V_MODEL_PATH="${E2V_MODEL_PATH:-${EXP_ROOT}/download/models/emotion2vec_plus_base/model.pt}"
export LOG_DIR="${LOG_DIR:-${EXP_ROOT}/paper_pipeline/slurm/logs}"

# --- HPC modules ---
module purge
module load GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0 FFmpeg/4.4.2

cd "${EXP_ROOT}"

# --- Activate virtualenv ---
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: venv not found at ${VENV_PATH}" >&2
fi

export PYTHONPATH="${EXP_ROOT}:${PYTHONPATH}"

export CB_CONFIG="${CB_CONFIG:-2x32}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  Paper Pipeline Environment"
echo "  EXP_ROOT:       ${EXP_ROOT}"
echo "  VENV_PATH:      ${VENV_PATH}"
echo "  DATA_ROOT:      ${DATA_ROOT}"
echo "  E2V_MODEL_PATH: ${E2V_MODEL_PATH}"
echo "  LOG_DIR:        ${LOG_DIR}"
echo "  CB_CONFIG:      ${CB_CONFIG}"
echo "  Python:         $(which python)"
echo "  Date:           $(date)"
echo "========================================"
