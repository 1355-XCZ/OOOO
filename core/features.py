"""
SSL Feature Extraction

Provides unified extractors for emotion2vec, HuBERT, and WavLM.
All extractors expose a .generate() API returning [{'feats': np.ndarray(T, D)}].

HuBERT/WavLM are loaded via HuggingFace transformers (replacing the legacy
s3prl dependency which is incompatible with torchaudio >= 2.1).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from .config import E2V_LOCAL_MODEL, SSL_FEATURE_DIMS

logger = logging.getLogger(__name__)


# ============================================================
# emotion2vec extractor
# ============================================================

def get_emotion2vec_extractor():
    """Load the emotion2vec feature extractor from local cache."""
    import warnings
    import os
    from funasr import AutoModel

    logger.info("Loading emotion2vec extractor from local cache...")

    prev_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    os.environ.setdefault('FUNASR_LOG_LEVEL', 'ERROR')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extractor = AutoModel(
            model=E2V_LOCAL_MODEL,
            disable_update=True,
            disable_log=True,
        )

    logging.root.setLevel(prev_level)
    logger.info("emotion2vec extractor loaded")
    return extractor


# ============================================================
# Feature extraction
# ============================================================

def extract_features(extractor, audio_path: str) -> Optional[torch.Tensor]:
    """Extract SSL features from a single audio file.

    Works with both funasr (emotion2vec) and TransformersExtractorWrapper.

    Returns:
        Tensor of shape [T, D] or None if extraction fails
    """
    try:
        result = extractor.generate(
            audio_path, output_dir=None, granularity="frame",
            extract_embedding=True, disable_pbar=True,
        )
        if result and len(result) > 0:
            feats = result[0].get('feats', None)
            if feats is None:
                for key in ['embedding', 'hidden_states', 'features']:
                    if key in result[0]:
                        feats = result[0][key]
                        break
            if feats is not None:
                if isinstance(feats, np.ndarray):
                    feats = torch.from_numpy(feats).float()
                elif isinstance(feats, list):
                    feats = torch.tensor(feats).float()
                return feats
    except Exception as e:
        logger.warning(f"Feature extraction failed: {audio_path}: {e}")
    return None


# ============================================================
# HuggingFace Transformers wrapper for HuBERT / WavLM
# ============================================================

SSL_HF_MODELS = {
    'hubert': 'facebook/hubert-large-ll60k',
    'wavlm': 'microsoft/wavlm-large',
}


class TransformersExtractorWrapper:
    """
    Wraps HuggingFace transformers models (HuBERT, WavLM) to provide the
    same .generate() API as funasr's emotion2vec.

    Usage:
        extractor = TransformersExtractorWrapper('hubert', device='cuda')
        result = extractor.generate(audio_path)
        feats = result[0]['feats']  # numpy array (T, D)
    """

    def __init__(self, ssl_model: str, device: str = 'cuda'):
        from transformers import AutoModel as HFAutoModel, AutoFeatureExtractor

        self.ssl_model = ssl_model
        self.device = device
        model_name = SSL_HF_MODELS[ssl_model]

        logger.info(f"Loading HuggingFace model: {model_name}")
        self.model = HFAutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        self._feature_dim = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers

        logger.info(f"Model loaded: {model_name} "
                     f"(dim={self._feature_dim}, layers={self._num_layers})")

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio to mono 16kHz."""
        wav, sr = torchaudio.load(str(audio_path))

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        return wav

    def extract(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file. Returns (T, D) numpy array."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav = self._load_audio(str(audio_path)).to(self.device)

        with torch.no_grad():
            outputs = self.model(wav, output_hidden_states=True)
            features = outputs.last_hidden_state.squeeze(0)  # (T, D)

        return features.cpu().numpy()

    def generate(self, audio_path: str, **kwargs) -> list:
        """
        Mimic funasr's .generate() API for compatibility with AudioFeatureDataset.

        Returns a list with one dict containing 'feats' key (numpy array (T, D)).
        """
        feats = self.extract(audio_path)
        return [{'feats': feats}]


# ============================================================
# Factory function
# ============================================================

def get_ssl_extractor(ssl_model: str, device: str = 'cuda'):
    """
    Factory function to get a feature extractor for the specified SSL model.

    All returned extractors have a .generate() method:
        result = extractor.generate(audio_path, output_dir=None, granularity="frame", extract_embedding=True)
        feats = result[0]['feats']  # numpy (T, D)

    Args:
        ssl_model: One of 'e2v', 'hubert', 'wavlm'
        device: 'cuda' or 'cpu'

    Returns:
        An extractor object with .generate() method.
    """
    if ssl_model == 'e2v':
        return get_emotion2vec_extractor()
    elif ssl_model in SSL_HF_MODELS:
        return TransformersExtractorWrapper(ssl_model, device=device)
    else:
        raise ValueError(
            f"Unknown SSL model: {ssl_model}. "
            f"Supported: {list(SSL_FEATURE_DIMS.keys())}"
        )


# ============================================================
# Codebook directory helper  (from ssl_feature_utils.py L156-176)
# ============================================================

def get_codebook_dir(base_codebook_dir: str, ssl_model: str, dataset: str,
                     config: str = '2x32') -> Path:
    """
    Get the codebook directory for a given SSL model, config, and dataset.

    Unified convention: codebooks/{ssl_model}/{config}/{dataset}/
    (All SSL models now use the same pattern.)

    Args:
        base_codebook_dir: Base codebook directory (e.g. codebooks/)
        ssl_model: One of 'e2v', 'hubert', 'wavlm'
        dataset: Dataset name (e.g. 'esd')
        config: Codebook config name (e.g. '2x32', '8x128')

    Returns:
        Path to the codebook directory
    """
    base = Path(base_codebook_dir)
    return base / ssl_model / config / dataset
