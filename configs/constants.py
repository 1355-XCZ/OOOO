"""
Single source of truth for all global constants.

Every magic number, default path, and shared parameter lives HERE.
All other modules import from this file — never hardcode these values elsewhere.
"""
import os
from pathlib import Path

_EXP_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Reproducibility
# ============================================================
GLOBAL_SEED: int = 1355

# ============================================================
# Training defaults
# ============================================================
DEFAULT_NUM_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_PATIENCE: int = 10
DEFAULT_MIN_DELTA: float = 0.0
DEFAULT_TRAIN_RATIO: float = 0.5
DEFAULT_VAL_RATIO: float = 0.1
DEFAULT_SAMPLES_PER_EMOTION: int = 500
DEFAULT_MAX_SAMPLES: int = 200

# ============================================================
# Paths  (override via environment variables or local_config.sh)
# ============================================================
DATA_ROOT: str = os.environ.get(
    'DATA_ROOT',
    str(_EXP_ROOT / 'download'),
)

E2V_MODEL_PATH: str = os.environ.get(
    'E2V_MODEL_PATH',
    str(_EXP_ROOT / 'download' / 'models' / 'emotion2vec_plus_base' / 'model.pt'),
)

E2V_MODEL_DIR: str = os.environ.get(
    'E2V_MODEL_DIR',
    os.path.dirname(E2V_MODEL_PATH),
)

# ============================================================
# Default codebook configuration
# ============================================================
DEFAULT_CB_CONFIG: str = '2x32'
