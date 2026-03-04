"""
Shared Training Utilities for RVQ Codebooks

Provides:
    - AudioFeatureDataset: dataset with in-memory SSL feature caching
    - collate_fn: batch collation with variable-length padding
    - train_codebook: full training loop with early stopping
    - create_balanced_sample: balanced sampling across emotions
"""

import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from .standard_rvq_official import StandardRVQOfficial, StandardRVQConfig
from configs.constants import GLOBAL_SEED
from configs.dataset_config import RVQConfig, TrainingConfig

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_stderr():
    """Temporarily redirect stderr to devnull to silence funasr's rtf_avg tqdm bars."""
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


# ============================================================
# Dataset  (from train_balanced_codebook.py L42-95)
# ============================================================

class AudioFeatureDataset(Dataset):
    """Audio feature dataset with in-memory caching.

    First epoch: extracts SSL features on the fly and caches in memory.
    Subsequent epochs: returns cached features directly (no re-extraction).
    """

    def __init__(self, audio_files: List[str], extractor, max_length: int = 500, feature_dim: int = 768):
        self.audio_files = audio_files
        self.extractor = extractor
        self.max_length = max_length
        self.feature_dim = feature_dim
        self._cache: Dict[int, dict] = {}

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        audio_path = self.audio_files[idx]

        try:
            with _suppress_stderr():
                result = self.extractor.generate(
                    audio_path,
                    output_dir=None,
                    granularity="frame",
                    extract_embedding=True,
                    disable_pbar=True,
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

                    if feats.dim() == 2 and feats.size(0) > self.max_length:
                        feats = feats[:self.max_length]

                    item = {
                        'features': feats,
                        'length': feats.size(0) if feats.dim() == 2 else 1,
                        'path': audio_path
                    }
                    self._cache[idx] = item
                    return item
        except Exception as e:
            logger.warning(f"Failed to extract features from {audio_path}: {e}")

        fallback = {
            'features': torch.zeros(1, self.feature_dim),
            'length': 0,
            'path': audio_path
        }
        self._cache[idx] = fallback
        return fallback


# ============================================================
# Collate function  (from train_balanced_codebook.py L98-122)
# ============================================================

def collate_fn(batch):
    """Batch collation function."""
    batch = [b for b in batch if b['length'] > 0]

    if len(batch) == 0:
        return None

    max_len = max(b['features'].size(0) for b in batch)

    features_list = []
    lengths = []

    for b in batch:
        feat = b['features']
        T, D = feat.shape
        if T < max_len:
            pad = torch.zeros(max_len - T, D)
            feat = torch.cat([feat, pad], dim=0)
        features_list.append(feat)
        lengths.append(b['length'])

    return {
        'features': torch.stack(features_list),
        'lengths': torch.tensor(lengths),
    }


# ============================================================
# Training loop  (from train_balanced_codebook.py L179-383)
# ============================================================

def train_codebook(
    train_files: List[str],
    val_files: List[str],
    extractor,
    output_path: Path,
    rvq_config: RVQConfig,
    training_config: TrainingConfig,
    codebook_name: str = "balanced",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """Train a single EMA-based RVQ codebook.

    Codebook entries are updated via exponential moving average during
    forward passes. No gradient-based optimization is used for the codebook.

    Args:
        train_files:     List of audio file paths for training
        val_files:       List of audio file paths for validation
        extractor:       SSL feature extractor (.generate() API)
        output_path:     Where to save the best checkpoint
        rvq_config:      RVQ architecture config
        training_config: Training hyperparameters
        codebook_name:   Name to store in checkpoint metadata
        extra_metadata:  Optional dict merged into saved checkpoint (e.g. for mixed codebooks)

    Returns:
        Best validation loss (or training loss if no val data)
    """
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    train_dataset = AudioFeatureDataset(train_files, extractor, feature_dim=rvq_config.feature_dim)
    val_dataset = AudioFeatureDataset(val_files, extractor, feature_dim=rvq_config.feature_dim)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info(f"Creating RVQ model: {rvq_config.num_layers} layers, "
                f"{rvq_config.codebook_size} codes (EMA, decay={StandardRVQConfig.decay})")

    config = StandardRVQConfig(
        feature_dim=rvq_config.feature_dim,
        num_layers=rvq_config.num_layers,
        codebook_size=rvq_config.codebook_size,
        use_cosine_sim=rvq_config.use_cosine_sim
    )
    model = StandardRVQOfficial(config).to(device)

    logger.info("Initializing codebook with first batch...")
    with torch.no_grad():
        for batch in train_loader:
            if batch is None:
                continue
            features = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            B, T, D = features.shape
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            valid_mask = t_idx < lengths.unsqueeze(1)
            _ = model(features, valid_mask=valid_mask)
            break
    logger.info("Codebook initialized")

    # Training loop — EMA codebook updates happen inside model.forward()
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0
    patience = getattr(training_config, 'patience', 10)
    min_delta = getattr(training_config, 'min_delta', 0.0)
    has_val_data = len(val_loader.dataset) > 0

    if not has_val_data:
        logger.warning("Validation set is empty! Will use training loss for model selection.")

    def _save_model(epoch, loss_value):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': {
                'feature_dim': rvq_config.feature_dim,
                'num_layers': rvq_config.num_layers,
                'codebook_size': rvq_config.codebook_size,
                'use_cosine_sim': rvq_config.use_cosine_sim,
            },
            'model_type': 'standard',
            'epoch': epoch,
            'val_loss': loss_value,
            'codebook_name': codebook_name,
        }
        if extra_metadata:
            save_dict.update(extra_metadata)
        torch.save(save_dict, output_path)
        logger.info(f"Saved best model to {output_path}")

    for epoch in range(training_config.num_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        for batch in pbar:
            if batch is None:
                continue

            features = batch['features'].to(device)
            lengths = batch['lengths'].to(device)

            B, T, D = features.shape
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            valid_mask = t_idx < lengths.unsqueeze(1)
            mask = valid_mask.float().unsqueeze(-1)

            reconstructed, indices, commit_loss, stats = model(features, valid_mask=valid_mask)

            mse_num = ((reconstructed - features) ** 2 * mask).sum()
            mse_den = (mask.sum() * D).clamp_min(1.0)
            recon_loss = mse_num / mse_den

            train_losses.append(recon_loss.item())
            pbar.set_postfix({'loss': f'{recon_loss.item():.4f}'})

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                features = batch['features'].to(device)
                lengths = batch['lengths'].to(device)

                B, T, D = features.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                valid_mask = t_idx < lengths.unsqueeze(1)
                mask = valid_mask.float().unsqueeze(-1)

                reconstructed, indices, commit_loss, stats = model(features, valid_mask=valid_mask)

                mse_num = ((reconstructed - features) ** 2 * mask).sum()
                mse_den = (mask.sum() * D).clamp_min(1.0)
                recon_loss = mse_num / mse_den
                val_losses.append(recon_loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')

        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if has_val_data:
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                _save_model(epoch + 1, best_val_loss)
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (best_val_loss={best_val_loss:.4f})")
                    break
        else:
            if avg_train_loss < best_train_loss - min_delta:
                best_train_loss = avg_train_loss
                patience_counter = 0
                _save_model(epoch + 1, avg_train_loss)
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (best_train_loss={best_train_loss:.4f})")
                    break

    return best_val_loss if has_val_data else best_train_loss


# ============================================================
# Data sampling helpers  (from train_balanced_codebook.py L128-176)
# ============================================================

def create_balanced_sample(
    files_by_emotion: Dict[str, List[str]],
    total_samples: int,
    seed: int = GLOBAL_SEED,
) -> List[str]:
    """Create a balanced dataset by sampling equal amounts from each emotion.

    Args:
        files_by_emotion: {emotion: [file_paths]}
        total_samples: Total number of samples to select
        seed: Random seed

    Returns:
        Shuffled list of file paths with equal representation per emotion
    """
    random.seed(seed)

    emotions = list(files_by_emotion.keys())
    n_emotions = len(emotions)
    samples_per_emotion = total_samples // n_emotions

    logger.info(f"Creating balanced dataset: {samples_per_emotion} per emotion x {n_emotions} = {total_samples}")

    balanced_files = []

    for emotion in emotions:
        files = files_by_emotion[emotion]
        if len(files) >= samples_per_emotion:
            sampled = random.sample(files, samples_per_emotion)
        else:
            logger.warning(f"  {emotion}: only {len(files)} files, using all with repetition")
            sampled = files.copy()
            while len(sampled) < samples_per_emotion:
                sampled.extend(random.sample(files, min(len(files), samples_per_emotion - len(sampled))))

        balanced_files.extend(sampled)
        logger.info(f"  {emotion}: {len(sampled)} files")

    random.shuffle(balanced_files)
    return balanced_files
