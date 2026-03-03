"""
Unified Configuration for BiasedCodebookExp

Centralizes all constants, paths, and seed management that were previously
scattered across individual scripts. All values are copied from the original
scripts to ensure reproducibility.

Sources:
    - evaluate_comprehensive_2x32.py (lines 50-86)
    - configs/dataset_config.py
    - ssl_feature_utils.py (lines 24-34)
"""

import random
import sys
import numpy as np
import torch
from pathlib import Path

# Ensure constants can be imported
_EXP_ROOT = Path(__file__).resolve().parent.parent
if str(_EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXP_ROOT))

from configs.constants import GLOBAL_SEED, E2V_MODEL_DIR as _E2V_MODEL_DIR, E2V_MODEL_PATH as _E2V_MODEL_PATH

# ============================================================
# Global random seed  (sourced from configs.constants)
# ============================================================
DEFAULT_SEED = GLOBAL_SEED


def set_seed(seed: int = GLOBAL_SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For full determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Project paths
# ============================================================
EXP_ROOT = Path(__file__).resolve().parents[1]  # BiasedCodebookExp_v2/

# Default directories (relative to EXP_ROOT)
CODEBOOK_DIR = EXP_ROOT / 'codebooks'
RESULTS_DIR = EXP_ROOT / 'results'
DATA_DIR = EXP_ROOT / 'data'
SPLITS_DIR = DATA_DIR / 'splits'
CLASSIFIER_DIR = EXP_ROOT / 'classifiers'


# ============================================================
# Unified path helpers  (see CONVENTIONS.md for full spec)
# ============================================================

DEFAULT_CODEBOOK_CONFIG = '2x32'


def codebook_path(ssl_model: str, dataset: str, codebook_type: str = 'balanced',
                  config: str = DEFAULT_CODEBOOK_CONFIG) -> Path:
    """Return canonical codebook path: codebooks/{ssl}/{config}/{dataset}/{type}.pt"""
    return CODEBOOK_DIR / ssl_model / config / dataset / f'{codebook_type}.pt'


def codebook_dir(ssl_model: str, dataset: str,
                 config: str = DEFAULT_CODEBOOK_CONFIG) -> Path:
    """Return codebook directory: codebooks/{ssl}/{config}/{dataset}/"""
    return CODEBOOK_DIR / ssl_model / config / dataset


def classifier_path(ssl_model: str, dataset: str) -> Path:
    """Return classifier path: classifiers/{ssl}/{dataset}/best_model.pt"""
    return CLASSIFIER_DIR / ssl_model / dataset / 'best_model.pt'


def result_path(experiment: str, ssl_model: str, source: str,
                target: str = None) -> Path:
    """Return canonical result path.

    ID:  results/{experiment}/{ssl}/{source}_id.json
    OOD: results/{experiment}/{ssl}/{source}_to_{target}_ood.json
    """
    base = RESULTS_DIR / experiment / ssl_model
    if target is None:
        return base / f'{source}_id.json'
    else:
        return base / f'{source}_to_{target}_ood.json'


def ensure_parent(path: Path) -> Path:
    """Create parent directories and return path (for use before writing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================
# emotion2vec paths  (copied from evaluate_comprehensive_2x32.py L81-86)
# ============================================================
E2V_LOCAL_MODEL = _E2V_MODEL_DIR
E2V_HEAD_PATH = _E2V_MODEL_PATH

# Official emotion2vec 9-class labels  (from dataset_config.py L13-14)
E2V_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral',
              'other', 'sad', 'surprised', 'unknown']
E2V_LABEL_TO_IDX = {label: idx for idx, label in enumerate(E2V_LABELS)}

# ============================================================
# Fair 4-class evaluation  (copied from evaluate_comprehensive_2x32.py L54-79)
# ============================================================
FAIR_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
FAIR_E2V_INDICES = [0, 3, 4, 6]  # indices in E2V 9-class output

# Dataset label -> fair-4 label mapping
# (copied verbatim from evaluate_comprehensive_2x32.py L57-78)
DATASET_TO_FAIR_MAP = {
    'esd':     {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'esd_en':  {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'esd_zh':  {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'iemocap': {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'ravdess': {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'cremad':  {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'cremad_clear': {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'cremad_ambig': {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'emodb':   {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'msp':     {'Angry': 'angry', 'Happy': 'happy', 'Neutral': 'neutral', 'Sad': 'sad'},
    # CAMEO datasets -- all use CAMEO's unified label scheme
    'cameo_crema_d':        {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_cafe':           {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_emns':           {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_emozionalmente': {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_enterface':      {'anger': 'angry', 'happiness': 'happy', 'sadness': 'sad'},  # no neutral
    'cameo_jl_corpus':      {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_mesd':           {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_nemo':           {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_oreau':          {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_pavoque':        {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_ravdess':        {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_resd':           {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_subesco':        {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'savee':                {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'tess':                 {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'meld':                 {'anger': 'angry', 'joy': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'asvp_esd':             {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
}

# ============================================================
# Default evaluation layers  (from evaluate_comprehensive_2x32.py L52)
# ============================================================
DEFAULT_LAYERS = list(range(1, 33))

# ============================================================
# SSL model dimensions  (from ssl_feature_utils.py L24-28)
# ============================================================
SSL_FEATURE_DIMS = {
    'e2v': 768,
    'hubert': 1024,
    'wavlm': 1024,
}

# ============================================================
# Language metadata  (from various analysis scripts)
# ============================================================
DATASET_LANGUAGES = {
    'esd': 'CN/EN', 'esd_en': 'EN', 'esd_zh': 'ZH', 'iemocap': 'EN', 'ravdess': 'EN',
    'cremad': 'EN', 'emodb': 'DE', 'msp': 'EN',
    'cameo_cafe': 'FR', 'cameo_crema_d': 'EN', 'cameo_emns': 'EN',
    'cameo_emozionalmente': 'IT', 'cameo_enterface': 'EN',
    'cameo_jl_corpus': 'EN', 'cameo_mesd': 'ES',
    'cameo_nemo': 'PL', 'cameo_oreau': 'FR', 'cameo_pavoque': 'DE',
    'cameo_ravdess': 'EN', 'cameo_resd': 'RU', 'cameo_subesco': 'BN',
    'savee': 'EN', 'tess': 'EN', 'meld': 'EN',
    'asvp_esd': 'EN',
}

# CAMEO dataset list  (from dataset_config.py L595-599)
CAMEO_DATASETS = [
    'cameo_crema_d', 'cameo_cafe', 'cameo_emns', 'cameo_emozionalmente',
    'cameo_enterface', 'cameo_jl_corpus', 'cameo_mesd', 'cameo_nemo',
    'cameo_oreau', 'cameo_pavoque', 'cameo_ravdess', 'cameo_resd', 'cameo_subesco',
]


# ============================================================
# Dynamic-fair evaluation (variable class count per pair)
# ============================================================

# Codebook source -> {biased_file_stem: e2v_label}
CODEBOOK_BIASED_MAP = {
    'esd_en': {
        'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral',
        'sad': 'sad', 'surprise': 'surprised',
    },
    'iemocap': {
        'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
    },
    'ravdess': {
        'angry': 'angry', 'disgust': 'disgusted', 'fearful': 'fearful',
        'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprised': 'surprised',
    },
    'cremad': {
        'angry': 'angry', 'disgust': 'disgusted', 'fear': 'fearful',
        'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
    },
}

# Full dataset -> e2v label mapping (ALL emotions, not just 4-fair)
DATASET_TO_FULL_MAP = {
    'esd_en':  {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprise': 'surprised'},
    'esd_zh':  {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprise': 'surprised'},
    'iemocap': {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'ravdess': {'angry': 'angry', 'disgust': 'disgusted', 'fearful': 'fearful', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprised': 'surprised'},
    'cremad':  {'angry': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'msp':     {'Angry': 'angry', 'Happy': 'happy', 'Neutral': 'neutral', 'Sad': 'sad', 'Surprise': 'surprised'},
    'emodb':   {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_crema_d':        {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_emns':           {'anger': 'angry', 'disgust': 'disgusted', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_enterface':      {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_jl_corpus':      {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_ravdess':        {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_cafe':           {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_emozionalmente': {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_mesd':           {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_nemo':           {'anger': 'angry', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_oreau':          {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'cameo_pavoque':        {'anger': 'angry', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_resd':           {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'cameo_subesco':        {'anger': 'angry', 'disgust': 'disgusted', 'fear': 'fearful', 'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprised'},
    'savee':                {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'tess':                 {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'},
    'meld':                 {'anger': 'angry', 'joy': 'happy', 'neutral': 'neutral', 'sadness': 'sad'},
    'asvp_esd':             {'angry': 'angry', 'disgust': 'disgusted', 'fearful': 'fearful', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprised': 'surprised'},
}


def get_source_emotions(source: str):
    """Sorted e2v labels for a codebook source's available biased codebooks."""
    return sorted(set(CODEBOOK_BIASED_MAP[source].values()))


def get_source_e2v_indices(source: str):
    """E2V 9-class indices for a source's emotions."""
    return [E2V_LABEL_TO_IDX[e] for e in get_source_emotions(source)]


def get_dataset_e2v_emotions(dataset: str):
    """Sorted e2v labels available in a dataset."""
    full_map = DATASET_TO_FULL_MAP.get(dataset)
    if not full_map:
        return []
    return sorted(set(full_map.values()))


def get_emotion_intersection(source: str, test_ds: str):
    """Sorted intersection of source codebook emotions and test dataset emotions."""
    return sorted(set(get_source_emotions(source)) & set(get_dataset_e2v_emotions(test_ds)))


# ============================================================
# Test mode configuration
# ============================================================
# When --test is passed, these overrides ensure the pipeline completes
# in seconds rather than hours, for quick smoke-test verification.

TEST_OVERRIDES = {
    'num_epochs': 2,
    'batch_size': 4,
    'max_samples': 5,
    'samples_per_emotion': 5,
    'num_layers': 2,            # RVQ layers
    'codebook_size': 4,         # codes per layer
}

TEST_DATASET = 'ravdess'        # smallest standard dataset
TEST_LAYERS = [1, 2]            # minimal layer set for evaluation
TEST_RESULTS_DIR = EXP_ROOT / 'results' / '_test'


def add_test_flag(parser):
    """Add --test flag to any argparse parser.

    Usage in scripts:
        from core.config import add_test_flag, apply_test_mode
        parser = argparse.ArgumentParser(...)
        # ... add your arguments ...
        add_test_flag(parser)
        args = parser.parse_args()
        apply_test_mode(args)
    """
    parser.add_argument(
        '--test', action='store_true',
        help='Smoke-test mode: tiny data, few epochs, verifies pipeline runs end-to-end',
    )


def apply_test_mode(args):
    """Patch argparse namespace with test-mode overrides.

    Only modifies attributes that exist on the namespace and haven't been
    explicitly set by the user on the command line. Prints a banner so the
    user knows test mode is active.

    Returns True if test mode is active, False otherwise.
    """
    if not getattr(args, 'test', False):
        return False

    print("=" * 60)
    print("  TEST MODE -- smoke-test with minimal data/epochs")
    print("=" * 60)

    overrides_applied = []
    for key, value in TEST_OVERRIDES.items():
        if hasattr(args, key):
            old = getattr(args, key)
            setattr(args, key, value)
            overrides_applied.append(f"  {key}: {old} -> {value}")

    # Override layers if present
    if hasattr(args, 'layers') and isinstance(getattr(args, 'layers'), str):
        old = args.layers
        args.layers = ','.join(map(str, TEST_LAYERS))
        overrides_applied.append(f"  layers: {old} -> {args.layers}")

    # Override output dir to avoid polluting real results
    if hasattr(args, 'output_dir'):
        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        old = args.output_dir
        args.output_dir = str(TEST_RESULTS_DIR)
        overrides_applied.append(f"  output_dir: ...{Path(old).name} -> {TEST_RESULTS_DIR}")

    if overrides_applied:
        print("Overrides applied:")
        for line in overrides_applied:
            print(line)
        print("=" * 60)

    return True
