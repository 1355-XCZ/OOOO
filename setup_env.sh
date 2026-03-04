#!/usr/bin/env bash
# ============================================================
# BiasedCodebookExp_v2 -- One-command Environment Setup
#
# Usage:
#   bash setup_env.sh [OPTIONS]
#
# This script:
#   1. (Optional) Loads HPC modules if running on a cluster
#   2. Creates a Python virtual environment
#   3. Installs PyTorch + torchaudio (GPU or CPU)
#   4. Installs all Python dependencies from requirements.txt
#   5. Verifies the installation and generates local_config.sh
#
# Works on:
#   - HPC clusters with module system (Spartan, Slurm, etc.)
#   - Local Linux/macOS machines with CUDA
#   - CPU-only machines (use --cpu)
#
# Prerequisites:
#   - Python 3.10+ available on PATH
#   - (GPU mode) NVIDIA driver + CUDA toolkit
# ============================================================

set -euo pipefail

# --- Resolve script path (SCRIPT_DIR = BiasedCodebookExp_v2/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Default values ---
VENV_PATH=""
CUDA_VERSION="118"
PYTHON_CMD="python3"
CPU_ONLY=false
SKIP_MODULES=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-path)    VENV_PATH="$2";      shift 2 ;;
        --cuda-version) CUDA_VERSION="$2";   shift 2 ;;
        --python)       PYTHON_CMD="$2";     shift 2 ;;
        --cpu)          CPU_ONLY=true;       shift ;;
        --skip-modules) SKIP_MODULES=true;   shift ;;
        -h|--help)
            cat <<'HELPEOF'
Usage: bash setup_env.sh [OPTIONS]

Options:
  --venv-path PATH       Path for the virtual environment
                         (default: ./venvs/biased-codebook-env)
  --cuda-version VER     CUDA version for PyTorch wheel, e.g. 118, 121, 124
                         (default: 118)
  --cpu                  Install CPU-only PyTorch (no CUDA required)
  --python CMD           Python executable to use (default: python3)
  --skip-modules         Skip HPC module loading even if 'module' is available
  -h, --help             Show this help message

Examples:
  # HPC with modules (auto-detected):
  bash setup_env.sh --venv-path /scratch/venvs/my-env

  # Local machine with CUDA 12.1:
  bash setup_env.sh --cuda-version 121

  # CPU-only (laptop / CI):
  bash setup_env.sh --cpu

  # Use a specific Python:
  bash setup_env.sh --python python3.10
HELPEOF
            exit 0
            ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${VENV_PATH}" ]]; then
    VENV_PATH="${SCRIPT_DIR}/venvs/biased-codebook-env"
fi

# --- Determine PyTorch index URL ---
if [[ "${CPU_ONLY}" == true ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="cpu"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
    TORCH_LABEL="cu${CUDA_VERSION}"
fi

echo "========================================"
echo "  Environment Setup"
echo "  VENV_PATH:     ${VENV_PATH}"
echo "  TORCH_VARIANT: ${TORCH_LABEL}"
echo "  PYTHON:        ${PYTHON_CMD}"
echo "========================================"

# --- Step 0/5: Load HPC modules if available ---
_has_module_cmd() {
    type module &>/dev/null
}

if [[ "${SKIP_MODULES}" == false ]] && _has_module_cmd; then
    echo "[0/5] HPC module system detected, loading modules ..."
    module purge 2>/dev/null || true
    for mod in GCCcore/11.3.0 Python/3.10.4 GCC/11.3.0 CUDA/11.7.0 \
               cuDNN/8.4.1.50-CUDA-11.7.0 FFmpeg/4.4.2; do
        if module load "${mod}" 2>/dev/null; then
            echo "  loaded ${mod}"
        else
            echo "  [SKIP] ${mod} not available"
        fi
    done
else
    echo "[0/5] No HPC module system (or --skip-modules), skipping module load"
fi

# --- Step 1: Create virtual environment ---
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[1/5] Virtual environment already exists at ${VENV_PATH}"
else
    echo "[1/5] Creating virtual environment ..."
    mkdir -p "$(dirname "${VENV_PATH}")"
    "${PYTHON_CMD}" -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
echo "  Active Python: $(which python) ($(python --version 2>&1))"

# Verify Python version >= 3.10
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "${PY_VER}" | cut -d. -f1)
PY_MINOR=$(echo "${PY_VER}" | cut -d. -f2)
if [[ "${PY_MAJOR}" -lt 3 ]] || [[ "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 10 ]]; then
    echo "ERROR: Python >= 3.10 required, found ${PY_VER}"
    exit 1
fi

# --- Step 2: Upgrade pip ---
echo "[2/5] Upgrading pip, setuptools, wheel ..."
pip install --upgrade pip setuptools wheel --quiet

# --- Step 3: Install PyTorch + torchaudio ---
echo "[3/5] Installing PyTorch + torchaudio (${TORCH_LABEL}) ..."
pip install torch torchaudio --index-url "${TORCH_INDEX}"

# --- Step 4: Install remaining dependencies ---
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
echo "[4/5] Installing dependencies from requirements.txt ..."
pip install -r "${REQUIREMENTS}"

# --- Step 5: Verification ---
echo ""
echo "[5/5] Verifying installation ..."
echo "----------------------------------------"
python -c "
import sys

ok = True
warnings = []

def check(name, import_name=None):
    global ok
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  [OK]   {name:30s} {ver}')
    except ImportError as e:
        print(f'  [FAIL] {name:30s} {e}')
        ok = False

check('torch')
check('torchaudio')
check('numpy')
check('scikit-learn', 'sklearn')
check('tqdm')
check('matplotlib')
check('funasr')
check('soundfile')
check('vector-quantize-pytorch', 'vector_quantize_pytorch')
check('einops')
check('transformers')

import torch
if torch.cuda.is_available():
    print(f'  [OK]   {\"CUDA\":30s} {torch.cuda.get_device_name(0)}')
else:
    print(f'  [WARN] {\"CUDA\":30s} not available (GPU tasks require CUDA)')

if ok:
    print('\nAll dependencies installed successfully!')
else:
    print('\nSome dependencies FAILED. Check errors above.')
    sys.exit(1)
"

# --- Generate local_config.sh if not present ---
LOCAL_CFG="${SCRIPT_DIR}/local_config.sh"
if [[ ! -f "${LOCAL_CFG}" ]]; then
    echo ""
    echo "Generating local_config.sh ..."
    cat > "${LOCAL_CFG}" << LOCALEOF
#!/bin/bash
# Auto-generated by setup_env.sh on $(date -Iseconds)
# Edit the paths below to match your environment.

export VENV_PATH="${VENV_PATH}"

# Root directory containing all datasets (IEMOCAP, ESD, RAVDESS, etc.)
# export DATA_ROOT="/path/to/your/data"

# Path to the emotion2vec model checkpoint
# export E2V_MODEL_PATH="/path/to/emotion2vec_plus_base/model.pt"

# (Optional) Override SLURM log directory
# export LOG_DIR="/path/to/logs"
LOCALEOF
    echo "  Created: ${LOCAL_CFG}"
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Activate environment:"
echo "    source ${VENV_PATH}/bin/activate"
echo ""
echo "  Configure local paths:"
echo "    vim ${LOCAL_CFG}"
echo "========================================"
