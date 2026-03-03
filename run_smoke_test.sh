#!/usr/bin/env bash
# ============================================================
# Smoke Test -- CPU-only end-to-end pipeline verification
#
# Tests: module imports, pipeline CLI, training script arguments,
# and figure generation dry-run. No real datasets or GPU required.
#
# Usage:
#   bash run_smoke_test.sh              # run locally
#   sbatch paper_pipeline/slurm/smoke_test.slurm  # submit to SLURM
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PASS=0
FAIL=0
TOTAL=0

pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo "  [PASS] $1"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo "  [FAIL] $1"; }

echo "============================================================"
echo "  BiasedCodebookExp_v2 -- Smoke Test (CPU)"
echo "============================================================"

# ----------------------------------------------------------
# Step 1: Environment verification
# ----------------------------------------------------------
echo ""
echo "--- Step 1: Environment verification ---"
if python verify_env.py 2>&1; then
    pass "verify_env.py"
else
    fail "verify_env.py (see output above)"
fi

# ----------------------------------------------------------
# Step 2: Pipeline --list
# ----------------------------------------------------------
echo ""
echo "--- Step 2: Pipeline --list ---"
OUTPUT=$(python -m paper_pipeline.pipeline --list 2>&1)
echo "${OUTPUT}"
if echo "${OUTPUT}" | grep -q "RQ"; then
    pass "pipeline --list"
else
    fail "pipeline --list"
fi

# ----------------------------------------------------------
# Step 3: Pipeline --dry-run (all RQs)
# ----------------------------------------------------------
echo ""
echo "--- Step 3: Pipeline dry-run ---"
if python -m paper_pipeline.pipeline --rq all --dry-run 2>&1; then
    pass "pipeline --rq all --dry-run"
else
    fail "pipeline --rq all --dry-run"
fi

# ----------------------------------------------------------
# Step 4: Training script --help (balanced)
# ----------------------------------------------------------
echo ""
echo "--- Step 4: Training scripts --help ---"
if python scripts/train/train_balanced_codebook.py --help > /dev/null 2>&1; then
    pass "train_balanced_codebook.py --help"
else
    fail "train_balanced_codebook.py --help"
fi

if python scripts/train/train_biased_codebook.py --help > /dev/null 2>&1; then
    pass "train_biased_codebook.py --help"
else
    fail "train_biased_codebook.py --help"
fi

# ----------------------------------------------------------
# Step 5: Core module imports
# ----------------------------------------------------------
echo ""
echo "--- Step 5: Core module imports ---"
MODULES=(
    "configs.constants"
    "configs.dataset_config"
    "core.config"
    "core.features"
    "core.quantize"
    "core.training"
    "core.classify"
    "paper_pipeline.config"
    "paper_pipeline.pipeline"
    "paper_pipeline.evaluators.rq1_evaluate"
    "paper_pipeline.evaluators.rq2_1_matched_ser"
    "paper_pipeline.evaluators.rq2_3_entropy"
    "paper_pipeline.evaluators.rq2_ce"
    "paper_pipeline.evaluators.rq4_evaluate"
    "paper_pipeline.evaluators.rq4_compute_f1"
    "paper_pipeline.evaluators.rq4_ratio_evaluate"
    "paper_pipeline.evaluators.rq4_ratio_compute_f1"
    "paper_pipeline.figures.rq1"
    "paper_pipeline.figures.rq2_combined"
    "paper_pipeline.figures.rq3_ratio_ambiguity_figure"
    "paper_pipeline.figures.rq4"
)

for mod in "${MODULES[@]}"; do
    if python -c "import ${mod}" 2>/dev/null; then
        pass "import ${mod}"
    else
        fail "import ${mod}"
    fi
done

# ----------------------------------------------------------
# Step 6: Synthetic data training test (if time permits)
# ----------------------------------------------------------
echo ""
echo "--- Step 6: Synthetic training test ---"
TEST_DIR="${SCRIPT_DIR}/results/_smoke_test"
SPLITS_DIR="${TEST_DIR}/splits"
CODEBOOK_DIR="${TEST_DIR}/codebooks"

python -c "
import json, numpy as np, soundfile as sf
from pathlib import Path

test_dir = Path('${TEST_DIR}')
splits_dir = test_dir / 'splits' / 'ravdess'
audio_dir = test_dir / 'audio'

emotions = ['angry', 'happy', 'neutral', 'sad']
train, val, test = {}, {}, {}

for emo in emotions:
    emo_dir = audio_dir / emo
    emo_dir.mkdir(parents=True, exist_ok=True)
    train[emo], val[emo], test[emo] = [], [], []
    for i in range(10):
        wav_path = emo_dir / f'{emo}_{i:03d}.wav'
        sf.write(str(wav_path), np.random.randn(8000).astype(np.float32), 16000)
        path_str = str(wav_path)
        if i < 6:
            train[emo].append(path_str)
        elif i < 8:
            val[emo].append(path_str)
        else:
            test[emo].append(path_str)

splits_dir.mkdir(parents=True, exist_ok=True)
for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(splits_dir / f'{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
print(f'Created {4 * 10} synthetic wav files + splits')
"

if [ -f "${SPLITS_DIR}/ravdess/train.json" ]; then
    pass "Synthetic data generated"
else
    fail "Synthetic data generation"
fi

# Train a tiny codebook on synthetic data (CPU, ~30 seconds)
echo "  Training tiny codebook on synthetic data ..."
if python scripts/train/train_balanced_codebook.py \
    --dataset ravdess \
    --ssl-model e2v \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir "${CODEBOOK_DIR}" \
    --codebook-size 4 \
    --num-layers 2 \
    --num-epochs 2 \
    --batch-size 4 \
    --device cpu \
    --codebook-name balanced \
    2>&1 | tail -5; then
    
    CB_PATH="${CODEBOOK_DIR}/e2v/4x2/ravdess/balanced.pt"
    if [ -f "${CB_PATH}" ]; then
        pass "Balanced codebook training (CPU)"
    else
        fail "Codebook file not found at ${CB_PATH}"
    fi
else
    fail "Balanced codebook training crashed"
fi

# Cleanup
echo ""
echo "  Cleaning up smoke test data ..."
rm -rf "${TEST_DIR}"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  Smoke Test Results:  ${PASS}/${TOTAL} passed,  ${FAIL} failed"
echo "============================================================"

if [ "${FAIL}" -gt 0 ]; then
    echo "SMOKE TEST FAILED -- ${FAIL} test(s) did not pass"
    exit 1
else
    echo "SMOKE TEST PASSED -- pipeline is functional"
    exit 0
fi
