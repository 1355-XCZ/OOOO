#!/usr/bin/env python3
"""
Train a single "natural" codebook on Emilia EN pre-extracted SSL features.

Emilia is a large-scale natural speech dataset (no emotion labels).
Features are already extracted as .npy files with shape (T, dim).

This script reuses the core training loop from core/training.py but replaces
AudioFeatureDataset with NpyFeatureDataset to load pre-extracted features.
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import (
    GLOBAL_SEED, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_PATIENCE, EMILIA_FEATURES_ROOT,
)
from core.config import set_seed, SSL_FEATURE_DIMS, CODEBOOK_DIR
from core.features import get_codebook_dir
from core.training import (
    collate_fn,
    StandardRVQOfficial, StandardRVQConfig, EncoderDecoderRVQ,
)
from configs.dataset_config import RVQConfig, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMILIA_FEATURE_DIRS = {
    'e2v':    Path(EMILIA_FEATURES_ROOT) / 'emotion2vec' / 'EN',
    'hubert': Path(EMILIA_FEATURES_ROOT) / 'hubert_large_ll60k' / 'EN',
    'wavlm':  Path(EMILIA_FEATURES_ROOT) / 'wavlm_large' / 'EN',
}


class NpyFeatureDataset(Dataset):
    """Dataset that loads pre-extracted .npy feature files directly."""

    def __init__(self, npy_files: List[Path], max_length: int = 500, feature_dim: int = 768):
        self.npy_files = npy_files
        self.max_length = max_length
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        path = self.npy_files[idx]
        try:
            feats = np.load(str(path))
            feats = torch.from_numpy(feats).float()

            if feats.dim() == 2 and feats.size(0) > self.max_length:
                feats = feats[:self.max_length]

            return {
                'features': feats,
                'length': feats.size(0) if feats.dim() == 2 else 1,
                'path': str(path),
            }
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return {
                'features': torch.zeros(1, self.feature_dim),
                'length': 0,
                'path': str(path),
            }


def train_codebook_from_npy(
    train_files: List[Path],
    val_files: List[Path],
    output_path: Path,
    rvq_config: RVQConfig,
    training_config: TrainingConfig,
    codebook_name: str = "natural",
    model_type: str = "standard",
) -> float:
    """Train a single RVQ codebook from pre-extracted .npy features.

    Copied from core/training.train_codebook with AudioFeatureDataset
    replaced by NpyFeatureDataset.
    """
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    # Create datasets
    train_dataset = NpyFeatureDataset(train_files, feature_dim=rvq_config.feature_dim)
    val_dataset = NpyFeatureDataset(val_files, feature_dim=rvq_config.feature_dim)

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    # Create RVQ model
    logger.info(f"Creating RVQ model: {rvq_config.num_layers} layers, {rvq_config.codebook_size} codes, type={model_type}")

    config = StandardRVQConfig(
        feature_dim=rvq_config.feature_dim,
        num_layers=rvq_config.num_layers,
        codebook_size=rvq_config.codebook_size,
        use_cosine_sim=rvq_config.use_cosine_sim,
    )

    if model_type == 'encoder_decoder':
        model = EncoderDecoderRVQ(config).to(device)
    else:
        model = StandardRVQOfficial(config).to(device)

    # Initialize codebook (pass valid_mask to avoid padding contamination)
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

    # Check trainable parameters
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in learnable_params)
    logger.info(f"Trainable parameters: {n_params}")

    # Optimizer
    if n_params > 0:
        if model_type == 'encoder_decoder':
            optimizer = torch.optim.AdamW(learnable_params, lr=training_config.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.num_epochs)
        else:
            optimizer = torch.optim.Adam(learnable_params, lr=training_config.learning_rate)
            scheduler = None
    else:
        optimizer = None
        scheduler = None

    # Training loop
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
            'model_type': model_type,
            'epoch': epoch,
            'val_loss': loss_value,
            'codebook_name': codebook_name,
        }
        torch.save(save_dict, output_path)
        logger.info(f"Saved best model to {output_path}")

    for epoch in range(training_config.num_epochs):
        # Train
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

            loss = recon_loss + commit_loss.mean() * rvq_config.commitment_weight

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(learnable_params, 1.0)
                optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = np.mean(train_losses)

        if scheduler is not None:
            scheduler.step()

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

                loss = recon_loss + commit_loss.mean() * rvq_config.commitment_weight
                val_losses.append(loss.item())

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


def main():
    parser = argparse.ArgumentParser(description='Train natural codebook on Emilia EN features')
    parser.add_argument('--ssl-model', type=str, required=True, choices=['e2v', 'hubert', 'wavlm'])
    parser.add_argument('--codebook-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    feat_dir = EMILIA_FEATURE_DIRS[args.ssl_model]
    all_files = sorted(feat_dir.glob('*.npy'))
    logger.info(f"Found {len(all_files)} Emilia EN features for {args.ssl_model}")

    random.seed(args.seed)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    feature_dim = SSL_FEATURE_DIMS[args.ssl_model]
    cb_config = f'{args.codebook_size}x{args.num_layers}'
    output_dir = get_codebook_dir(str(CODEBOOK_DIR), args.ssl_model, 'emilia', config=cb_config)
    output_path = output_dir / 'natural.pt'

    logger.info("=" * 60)
    logger.info(f"Training Natural Codebook (Emilia EN)")
    logger.info(f"  SSL: {args.ssl_model} (dim={feature_dim})")
    logger.info(f"  Config: {cb_config}")
    logger.info(f"  Train: {len(train_files)}, Val: {len(val_files)}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    rvq_config = RVQConfig(
        feature_dim=feature_dim,
        num_layers=args.num_layers,
        codebook_size=args.codebook_size,
    )
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )

    best_loss = train_codebook_from_npy(
        train_files, val_files, output_path, rvq_config, training_config,
    )

    logger.info("=" * 60)
    logger.info(f"Done! Best val loss: {best_loss:.4f}")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
