"""
Dataset Configuration for Biased Codebook Experiment

Dataset configuration and emotion mapping to official emotion2vec 9 classes:
0: angry, 1: disgusted, 2: fearful, 3: happy, 4: neutral, 5: other, 6: sad, 7: surprised, 8: unknown
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from configs.constants import (
    GLOBAL_SEED, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE, DEFAULT_MIN_DELTA, DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO,
    DATA_ROOT, E2V_MODEL_PATH as _E2V_MODEL_PATH,
)

# Official emotion2vec 9-class labels
E2V_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
E2V_LABEL_TO_IDX = {label: idx for idx, label in enumerate(E2V_LABELS)}

# Re-export for backward compatibility
E2V_MODEL_PATH = _E2V_MODEL_PATH

_DATA_ROOT = DATA_ROOT


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    data_root: str
    emotions: List[str]  # dataset's original emotion labels
    emotion_to_e2v: Dict[str, str]  # dataset emotion -> emotion2vec 9-class mapping
    enabled: bool = True  # whether this dataset is enabled
    min_samples_per_emotion: int = 50  # minimum samples per emotion
    
    @property
    def e2v_emotions(self) -> List[str]:
        """Return mapped emotion2vec emotion list (deduplicated)."""
        return list(set(self.emotion_to_e2v.values()))
    
    @property
    def num_emotions(self) -> int:
        """Return the number of valid emotions."""
        return len(self.emotions)


# ============================================================
# Dataset configurations
# ============================================================

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    'esd_en': DatasetConfig(
        name='ESD-English',
        data_root=f'{_DATA_ROOT}/ESD/Emotion Speech Dataset',
        emotions=['angry', 'happy', 'neutral', 'sad', 'surprise'],
        emotion_to_e2v={
            'angry': 'angry',       # 0
            'happy': 'happy',       # 3
            'neutral': 'neutral',   # 4
            'sad': 'sad',           # 6
            'surprise': 'surprised' # 7
        },
        enabled=True,
    ),

    'iemocap': DatasetConfig(
        name='IEMOCAP',
        data_root=f'{_DATA_ROOT}/IEMOCAP_full_release',
        emotions=['angry', 'happy', 'neutral', 'sad'],  # merged hap+exc into happy
        emotion_to_e2v={
            'angry': 'angry',       # 0
            'happy': 'happy',       # 3 (includes excited)
            'neutral': 'neutral',   # 4
            'sad': 'sad',           # 6
        },
        enabled=True,
    ),
    
    'ravdess': DatasetConfig(
        name='RAVDESS',
        data_root=f'{_DATA_ROOT}/RAVDESS',
        emotions=['angry', 'happy', 'neutral', 'sad', 'fearful', 'disgust', 'surprised'],
        emotion_to_e2v={
            'angry': 'angry',       # 0
            'happy': 'happy',       # 3
            'neutral': 'neutral',   # 4
            'sad': 'sad',           # 6
            'fearful': 'fearful',   # 2
            'disgust': 'disgusted', # 1
            'surprised': 'surprised' # 7
            # Note: calm is skipped (not in emotion2vec 9 classes)
        },
        enabled=True,
    ),
    
    'cremad': DatasetConfig(
        name='CREMA-D',
        data_root=f'{_DATA_ROOT}/CREMA-D',
        emotions=['angry', 'happy', 'neutral', 'sad', 'fear', 'disgust'],
        emotion_to_e2v={
            'angry': 'angry',       # 0
            'happy': 'happy',       # 3
            'neutral': 'neutral',   # 4
            'sad': 'sad',           # 6
            'fear': 'fearful',      # 2
            'disgust': 'disgusted', # 1
        },
        enabled=True,
    ),
    'cremad_clear': DatasetConfig(
        name='CREMA-D (clear, 100% agreement)',
        data_root=f'{_DATA_ROOT}/CREMA-D',
        emotions=['angry', 'happy', 'neutral', 'sad'],
        emotion_to_e2v={
            'angry': 'angry',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
        },
        enabled=True,
    ),
    'cremad_ambig': DatasetConfig(
        name='CREMA-D (ambiguous, <100% agreement)',
        data_root=f'{_DATA_ROOT}/CREMA-D',
        emotions=['angry', 'happy', 'neutral', 'sad'],
        emotion_to_e2v={
            'angry': 'angry',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
        },
        enabled=True,
    ),
    
    'msp': DatasetConfig(
        name='MSP-Podcast',
        data_root=f'{_DATA_ROOT}/MSP',
        emotions=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'],
        emotion_to_e2v={
            'Angry': 'angry',       # 0
            'Happy': 'happy',       # 3
            'Neutral': 'neutral',   # 4
            'Sad': 'sad',           # 6
            'Surprise': 'surprised' # 7
        },
        enabled=True,
    ),

    # ============================================================
    # CAMEO OOD test datasets (amu-cai/CAMEO on HuggingFace)
    # Directory structure: CAMEO/{split_name}/{emotion}/*.wav
    # ============================================================

    'cameo_emns': DatasetConfig(
        name='CAMEO-EMNS',
        data_root=f'{_DATA_ROOT}/CAMEO/emns',
        emotions=['anger', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
            # excitement -> debatable, not mapped for now
            # sarcasm -> cannot be mapped
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    'cameo_enterface': DatasetConfig(
        name='CAMEO-EnterFace',
        data_root=f'{_DATA_ROOT}/CAMEO/enterface',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'sadness': 'sad',
            'surprise': 'surprised',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # JL Corpus (English) -- only core emotions mapped, others skipped
    'cameo_jl_corpus': DatasetConfig(
        name='CAMEO-JL-Corpus',
        data_root=f'{_DATA_ROOT}/CAMEO/jl_corpus',
        emotions=['anger', 'happiness', 'neutral', 'sadness'],
        emotion_to_e2v={
            'anger': 'angry',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            # anxiety, apology, assertiveness, concern, encouragement, excitement -> cannot be mapped
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

}


def get_enabled_datasets() -> Dict[str, DatasetConfig]:
    """Get all enabled dataset configurations."""
    return {k: v for k, v in DATASET_CONFIGS.items() if v.enabled}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get configuration for the specified dataset."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]


# ============================================================
# Data file pattern matching rules
# ============================================================

DATASET_FILE_PATTERNS = {
    'esd': {
        'pattern': '**/*.wav',
        'emotion_from_path': lambda p: p.parent.name.lower(),
    },
    'iemocap': {
        'pattern': '**/*.wav',
        'annotation_dir': 'EmoEvaluation',
    },
    'ravdess': {
        'pattern': '**/*.wav',
        'emotion_code_map': {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        },
    },
    'cremad': {
        'pattern': '**/*.wav',
        'emotion_code_map': {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        },
    },
    'msp': {
        'json_path': f'{_DATA_ROOT}/MSP/msp_ambigous.json',
    },
}

CAMEO_DATASETS = ['cameo_emns', 'cameo_enterface', 'cameo_jl_corpus']

CAMEO_LANGUAGES = {
    'cameo_emns': 'English',
    'cameo_enterface': 'English',
    'cameo_jl_corpus': 'English',
}


# ============================================================
# RVQ configuration
# ============================================================

@dataclass
class RVQConfig:
    """RVQ codebook configuration."""
    feature_dim: int = 768  # emotion2vec feature dimension
    num_layers: int = 32    # number of RVQ layers
    codebook_size: int = 2  # codebook size per layer
    use_cosine_sim: bool = False
    decay: float = 0.99
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 10
    threshold_ema_dead_code: float = 2.0

    @property
    def config_name(self) -> str:
        """Return config tag as codebook_size x num_layers, e.g. '2x32', '128x8'."""
        return f'{self.codebook_size}x{self.num_layers}'


# Default RVQ configuration
DEFAULT_RVQ_CONFIG = RVQConfig()


# ============================================================
# Training configuration
# ============================================================

@dataclass 
class TrainingConfig:
    """Training configuration — defaults sourced from configs.constants."""
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    device: str = 'cuda'
    seed: int = GLOBAL_SEED
    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA

    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO


DEFAULT_TRAINING_CONFIG = TrainingConfig()
