"""
SSL Feature Extraction

Provides unified extractors for emotion2vec, HuBERT, and WavLM.
All extractors expose a .generate() API returning [{'feats': np.ndarray(T, D)}].

HuBERT/WavLM use s3prl (fairseq-based) to ensure feature-level
reproducibility with the original HuBERT/WavLM checkpoints.
A torchaudio compatibility shim is applied so that s3prl works
with torchaudio >= 2.1 where legacy APIs have been removed.
"""

import logging
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

# ---- torchaudio compatibility shim for s3prl ----
# s3prl internally imports deprecated / removed torchaudio sub-modules.
# We inject lightweight stubs so that s3prl loads cleanly on any version.

if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda *_a, **_kw: None

if 'torchaudio._backend' not in sys.modules:
    _backend_stub = types.ModuleType('torchaudio._backend')
    _backend_stub.set_audio_backend = lambda *_a, **_kw: None
    sys.modules['torchaudio._backend'] = _backend_stub
    torchaudio._backend = _backend_stub

if 'torchaudio.sox_effects' not in sys.modules:
    _sox_stub = types.ModuleType('torchaudio.sox_effects')
    _sox_stub.effect_names = lambda: []
    _sox_stub.apply_effects_tensor = lambda tensor, sr, effects, channels_first=True: (tensor, sr)
    sys.modules['torchaudio.sox_effects'] = _sox_stub
    torchaudio.sox_effects = _sox_stub
# ---- end shim ----

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

    Works with both funasr (emotion2vec) and S3PRLExtractorWrapper (HuBERT, WavLM).

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
# S3PRL wrapper for HuBERT / WavLM
# ============================================================

SSL_TO_S3PRL = {
    'hubert': 'hubert_large_ll60k',
    'wavlm': 'wavlm_large',
}


class S3PRLExtractorWrapper:
    """
    Wraps s3prl upstream models to provide the same .generate() API
    as funasr's emotion2vec, so downstream code is backend-agnostic.

    Usage:
        extractor = S3PRLExtractorWrapper('hubert_large_ll60k', device='cuda')
        result = extractor.generate(audio_path)
        feats = result[0]['feats']  # numpy array (T, D)
    """

    def __init__(self, upstream_name: str, device: str = 'cuda'):
        self.upstream_name = upstream_name
        self.device = device

        try:
            from s3prl.nn import S3PRLUpstream
        except ImportError:
            raise ImportError(
                "s3prl is not installed. Run: pip install s3prl\n"
                "Or see: https://github.com/s3prl/s3prl"
            )

        logger.info(f"Loading s3prl model: {upstream_name}")
        self.model = S3PRLUpstream(upstream_name).to(device)
        self.model.eval()

        with torch.no_grad():
            dummy_wav = torch.randn(1, 16000).to(device)
            dummy_len = torch.LongTensor([16000]).to(device)
            hidden_states, output_lens = self.model(dummy_wav, dummy_len)
            self._feature_dim = hidden_states[-1].shape[-1]

        logger.info(f"s3prl model loaded: {upstream_name} "
                     f"(dim={self._feature_dim}, layers={len(hidden_states)})")

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
        wav_len = torch.LongTensor([wav.shape[1]]).to(self.device)

        with torch.no_grad():
            hidden_states, output_lens = self.model(wav, wav_len)
            features = hidden_states[-1].squeeze(0)  # (T, D)

        return features.cpu().numpy()

    def generate(self, audio_path: str, **kwargs) -> list:
        """
        Mimic funasr's .generate() API for compatibility.

        Returns a list with one dict containing 'feats' key (numpy array (T, D)).
        Extra kwargs (output_dir, granularity, extract_embedding) are accepted but ignored.
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
        result = extractor.generate(audio_path)
        feats = result[0]['feats']  # numpy (T, D)

    Args:
        ssl_model: One of 'e2v', 'hubert', 'wavlm'
        device: 'cuda' or 'cpu'
    """
    if ssl_model == 'e2v':
        return get_emotion2vec_extractor()
    elif ssl_model in SSL_TO_S3PRL:
        upstream_name = SSL_TO_S3PRL[ssl_model]
        return S3PRLExtractorWrapper(upstream_name, device=device)
    else:
        raise ValueError(
            f"Unknown SSL model: {ssl_model}. "
            f"Supported: {list(SSL_FEATURE_DIMS.keys())}"
        )


# ============================================================
# Codebook directory helper
# ============================================================

def get_codebook_dir(base_codebook_dir: str, ssl_model: str, dataset: str,
                     config: str = '2x32') -> Path:
    """
    Get the codebook directory for a given SSL model, config, and dataset.

    Convention: codebooks/{ssl_model}/{config}/{dataset}/
    """
    base = Path(base_codebook_dir)
    return base / ssl_model / config / dataset
