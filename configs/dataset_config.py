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
    'esd': DatasetConfig(
        name='ESD',
        data_root=f'{_DATA_ROOT}/ESD/Emotion Speech Dataset',
        emotions=['angry', 'happy', 'neutral', 'sad', 'surprise'],
        emotion_to_e2v={
            'angry': 'angry',       # 0
            'happy': 'happy',       # 3
            'neutral': 'neutral',   # 4
            'sad': 'sad',           # 6
            'surprise': 'surprised' # 7 (note spelling difference)
        },
        enabled=True,
    ),

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
    
    'esd_zh': DatasetConfig(
        name='ESD-Chinese',
        data_root=f'{_DATA_ROOT}/ESD/Emotion Speech Dataset',
        emotions=['angry', 'happy', 'neutral', 'sad', 'surprise'],
        emotion_to_e2v={
            'angry': 'angry',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'surprise': 'surprised'
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
    
    'emodb': DatasetConfig(
        name='EmoDB',
        data_root=f'{_DATA_ROOT}/EmoDB',
        emotions=['anger', 'happiness', 'neutral', 'sadness', 'fear', 'disgust'],
        emotion_to_e2v={
            'anger': 'angry',       # 0
            'happiness': 'happy',   # 3
            'neutral': 'neutral',   # 4
            'sadness': 'sad',       # 6
            'fear': 'fearful',      # 2
            'disgust': 'disgusted', # 1
            # Note: boredom is skipped (not in emotion2vec 9 classes)
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
    # CAMEO multilingual datasets (amu-cai/CAMEO)
    # Used for OOD testing only, not for training
    # Directory structure: CAMEO/{split_name}/{emotion}/*.wav
    # ============================================================

    # CREMA-D (English) -- CAMEO version (differs from original cremad data, avoids conflict)
    'cameo_crema_d': DatasetConfig(
        name='CAMEO-CREMA-D',
        data_root=f'{_DATA_ROOT}/CAMEO/crema_d',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # CaFE (French)
    'cameo_cafe': DatasetConfig(
        name='CAMEO-CaFE',
        data_root=f'{_DATA_ROOT}/CAMEO/cafe',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # EMNS (English) -- sarcasm cannot be mapped, skipped
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

    # Emozionalmente (Italian)
    'cameo_emozionalmente': DatasetConfig(
        name='CAMEO-Emozionalmente',
        data_root=f'{_DATA_ROOT}/CAMEO/emozionalmente',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # EnterFace (English)
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

    # MESD (Spanish)
    'cameo_mesd': DatasetConfig(
        name='CAMEO-MESD',
        data_root=f'{_DATA_ROOT}/CAMEO/mesd',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # NEMO (Polish)
    'cameo_nemo': DatasetConfig(
        name='CAMEO-NEMO',
        data_root=f'{_DATA_ROOT}/CAMEO/nemo',
        emotions=['anger', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # OREAU (French)
    'cameo_oreau': DatasetConfig(
        name='CAMEO-OREAU',
        data_root=f'{_DATA_ROOT}/CAMEO/oreau',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # Pavoque (German) -- poker cannot be mapped, skipped
    'cameo_pavoque': DatasetConfig(
        name='CAMEO-Pavoque',
        data_root=f'{_DATA_ROOT}/CAMEO/pavoque',
        emotions=['anger', 'happiness', 'neutral', 'sadness'],
        emotion_to_e2v={
            'anger': 'angry',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            # poker -> cannot be mapped
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # RAVDESS (English) -- CAMEO version, calm skipped
    'cameo_ravdess': DatasetConfig(
        name='CAMEO-RAVDESS',
        data_root=f'{_DATA_ROOT}/CAMEO/ravdess',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
            # calm -> cannot be mapped
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # RESD (Russian) -- enthusiasm cannot be mapped, skipped
    'cameo_resd': DatasetConfig(
        name='CAMEO-RESD',
        data_root=f'{_DATA_ROOT}/CAMEO/resd',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            # enthusiasm -> cannot be mapped
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # SUBESCO (Bengali)
    'cameo_subesco': DatasetConfig(
        name='CAMEO-SUBESCO',
        data_root=f'{_DATA_ROOT}/CAMEO/subesco',
        emotions=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
        emotion_to_e2v={
            'anger': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happiness': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprised',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # ================================================================
    # Additional OOD Test Datasets (English, 4-fair emotions)
    # ================================================================

    # SAVEE (English) -- 4 speakers, filename: {speaker}_{emotion_code}{number}.wav
    # Resampled from 44100Hz to 16kHz
    'savee': DatasetConfig(
        name='SAVEE',
        data_root=f'{_DATA_ROOT}/SAVEE_16k',
        emotions=['angry', 'happy', 'neutral', 'sad'],
        emotion_to_e2v={
            'angry': 'angry',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # TESS (English) -- 2 speakers (OAF, YAF), folders: {speaker}_{emotion}
    # Resampled from 24414Hz to 16kHz
    'tess': DatasetConfig(
        name='TESS',
        data_root=f'{_DATA_ROOT}/TESS_16k',
        emotions=['angry', 'happy', 'neutral', 'sad'],
        emotion_to_e2v={
            'angry': 'angry',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # MELD (English) -- Friends TV series utterances, organized by emotion after extraction
    'meld': DatasetConfig(
        name='MELD',
        data_root=f'{_DATA_ROOT}/MELD/audio',
        emotions=['anger', 'joy', 'neutral', 'sadness'],
        emotion_to_e2v={
            'anger': 'angry',
            'joy': 'happy',
            'neutral': 'neutral',
            'sadness': 'sad',
        },
        enabled=True,
        min_samples_per_emotion=50,
    ),

    # ASVP-ESD (multilingual, speech + non-speech emotional sounds)
    # Filename encoding: field 3 = emotion code
    # Emotion codes: 02=neutral, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    # Skipped: 01=boredom, 09=excited, 10=pleasure, 11=pain, 12=disappointment, 13=breath
    'asvp_esd': DatasetConfig(
        name='ASVP-ESD',
        data_root=f'{_DATA_ROOT}/ASVP-ESD/ASVP-ESD-Update/Audio',
        emotions=['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
        emotion_to_e2v={
            'neutral': 'neutral',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fearful',
            'disgust': 'disgusted',
            'surprised': 'surprised',
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
        # ESD directory structure: {data_root}/{speaker_id}/{emotion}/*.wav
        'pattern': '**/*.wav',
        'emotion_from_path': lambda p: p.parent.name.lower(),  # get emotion from parent directory
    },
    'iemocap': {
        # IEMOCAP requires reading from annotation files
        'pattern': '**/*.wav',
        'annotation_dir': 'EmoEvaluation',
    },
    'ravdess': {
        # RAVDESS filename format: XX-XX-XX-XX-XX-XX-XX.wav, 3rd field is emotion
        'pattern': '**/*.wav',
        'emotion_code_map': {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        },
    },
    'cremad': {
        # CREMA-D filename format: XXXX_XXX_EMO_XX.wav
        'pattern': '**/*.wav',
        'emotion_code_map': {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        },
    },
    'emodb': {
        # EmoDB filename: last letter is the emotion code
        'pattern': '**/*.wav',
        'emotion_code_map': {
            'W': 'anger', 'L': 'boredom', 'E': 'disgust',
            'A': 'fear', 'F': 'happiness', 'N': 'neutral', 'T': 'sadness'
        },
    },
    'msp': {
        # MSP reads from JSON file
        'json_path': f'{_DATA_ROOT}/MSP/msp_ambigous.json',
    },
}

# CAMEO unified file pattern: {data_root}/{emotion}/*.wav
CAMEO_DATASETS = [
    'cameo_crema_d', 'cameo_cafe', 'cameo_emns', 'cameo_emozionalmente',
    'cameo_enterface', 'cameo_jl_corpus', 'cameo_mesd', 'cameo_nemo',
    'cameo_oreau', 'cameo_pavoque', 'cameo_ravdess', 'cameo_resd', 'cameo_subesco',
]

# CAMEO language information
CAMEO_LANGUAGES = {
    'cameo_crema_d': 'English',
    'cameo_cafe': 'French',
    'cameo_emns': 'English',
    'cameo_emozionalmente': 'Italian',
    'cameo_enterface': 'English',
    'cameo_jl_corpus': 'English',
    'cameo_mesd': 'Spanish',
    'cameo_nemo': 'Polish',
    'cameo_oreau': 'French',
    'cameo_pavoque': 'German',
    'cameo_ravdess': 'English',
    'cameo_resd': 'Russian',
    'cameo_subesco': 'Bengali',
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
