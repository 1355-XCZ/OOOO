#!/usr/bin/env python3
"""
Train Ambiguity-Level Codebook

Trains codebooks from samples grouped by annotator agreement levels:
  - high (100% consistency)
  - mid  (~80% average consistency)
  - low  (~67% average consistency)

Supported datasets: iemocap, cremad.
4 emotions (angry/happy/neutral/sad) x 3 levels = 12 codebooks per dataset.
152 samples per codebook, codebook_size=2, num_layers=32.

Usage:
  python train_ambiguity_codebook.py --dataset iemocap --emotion angry --level high
  python train_ambiguity_codebook.py --dataset cremad --emotion sad --level low
"""

import json
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import (
    GLOBAL_SEED, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
)
from core.config import set_seed, SSL_FEATURE_DIMS, CODEBOOK_DIR, SPLITS_DIR
from core.features import get_ssl_extractor, get_codebook_dir
from core.training import AudioFeatureDataset, collate_fn
from core.standard_rvq_official import StandardRVQOfficial, StandardRVQConfig, EncoderDecoderRVQ
from configs.dataset_config import RVQConfig, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Training (ambiguity-specific: extra_metadata, training_info)
# ============================================================

def train_codebook(
    train_files: List[str],
    val_files: List[str],
    extractor,
    output_path: Path,
    rvq_config: RVQConfig,
    training_config: TrainingConfig,
    codebook_name: str = "ambiguity",
    extra_metadata: Dict = None,
    model_type: str = "standard"
):
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    train_dataset = AudioFeatureDataset(train_files, extractor, max_length=3000, feature_dim=rvq_config.feature_dim)
    val_dataset = AudioFeatureDataset(val_files, extractor, max_length=3000, feature_dim=rvq_config.feature_dim)

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    logger.info(f"Creating RVQ model: {rvq_config.num_layers} layers, {rvq_config.codebook_size} codes, dim={rvq_config.feature_dim}, type={model_type}")

    config = StandardRVQConfig(
        feature_dim=rvq_config.feature_dim,
        num_layers=rvq_config.num_layers,
        codebook_size=rvq_config.codebook_size,
        use_cosine_sim=rvq_config.use_cosine_sim
    )
    
    if model_type == 'encoder_decoder':
        model = EncoderDecoderRVQ(config).to(device)
    else:
        model = StandardRVQOfficial(config).to(device)

    logger.info("Initializing codebook with first batch...")
    with torch.no_grad():
        for batch in train_loader:
            if batch is not None:
                features = batch['features'].to(device)
                lengths = batch['lengths'].to(device)
                B, T, D = features.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                valid_mask = t_idx < lengths.unsqueeze(1)
                _ = model(features, valid_mask=valid_mask)
                break
    logger.info("Codebook initialized")

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in learnable_params)
    logger.info(f"Trainable parameters: {n_params}")

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

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

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

        logger.info(
            f"Epoch {epoch+1}/{training_config.num_epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'feature_dim': rvq_config.feature_dim,
                    'num_layers': rvq_config.num_layers,
                    'codebook_size': rvq_config.codebook_size,
                    'use_cosine_sim': rvq_config.use_cosine_sim,
                    'model_type': model_type,
                },
                'model_type': model_type,
                'training_info': {
                    'best_val_loss': best_val_loss,
                    'best_epoch': epoch + 1,
                    'total_epochs': training_config.num_epochs,
                    'codebook_name': codebook_name,
                    'num_train_files': len(train_files),
                    'num_val_files': len(val_files),
                },
            }
            if extra_metadata:
                save_dict['extra_metadata'] = extra_metadata

            torch.save(save_dict, str(output_path))
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f}) to {output_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    return best_val_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train ambiguity-level codebook')
    parser.add_argument('--dataset', type=str, default='iemocap',
                        choices=['iemocap', 'cremad'],
                        help='Source dataset for ambiguity splits')
    parser.add_argument('--emotion', type=str, required=True,
                        choices=['angry', 'happy', 'neutral', 'sad'],
                        help='Target emotion')
    parser.add_argument('--level', type=str, required=True,
                        choices=['high', 'mid', 'low'],
                        help='Ambiguity level (high=100%%, mid=80%%, low=67%%)')
    parser.add_argument('--splits-file', type=str, default=None,
                        help='Path to ambiguity_splits.json (auto-resolved from --dataset if omitted)')
    parser.add_argument('--ssl-model', type=str, default='e2v',
                        choices=['e2v', 'hubert', 'wavlm'],
                        help='SSL model for feature extraction')
    parser.add_argument('--output-dir', type=str,
                        default=str(CODEBOOK_DIR),
                        help='Base codebook directory')
    parser.add_argument('--codebook-config', type=str, default=None,
                        help='Codebook config tag (e.g. 2x32). Auto-derived from num-layers/codebook-size if omitted.')
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--num-layers', type=int, default=32, help='RVQ layers')
    parser.add_argument('--codebook-size', type=int, default=2, help='Codes per layer')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'encoder_decoder'],
                        help='Model architecture: standard (direct RVQ) or encoder_decoder (encoder-RVQ-decoder)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load ambiguity splits
    splits_file = args.splits_file or str(SPLITS_DIR / args.dataset / 'ambiguity_splits.json')
    with open(splits_file) as f:
        splits = json.load(f)

    key = f"{args.emotion}_{args.level}"
    if key not in splits:
        raise ValueError(f"Key '{key}' not found in ambiguity_splits.json. Available: {list(splits.keys())}")

    split_info = splits[key]
    all_files = split_info['files']
    target_consistency = split_info['target_consistency']
    actual_consistency = split_info['actual_consistency']

    feature_dim = SSL_FEATURE_DIMS[args.ssl_model]

    logger.info("=" * 60)
    logger.info(f"Training Ambiguity Codebook")
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  Emotion:    {args.emotion}")
    logger.info(f"  Level:      {args.level}")
    logger.info(f"  SSL Model:  {args.ssl_model} (dim={feature_dim})")
    logger.info(f"  Target consistency: {target_consistency:.2%}")
    logger.info(f"  Actual consistency: {actual_consistency:.4f}")
    logger.info(f"  Total files: {len(all_files)}")
    logger.info(f"  RVQ: {args.num_layers} layers x {args.codebook_size} codes")
    logger.info("=" * 60)

    # Train/Val split
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * args.val_ratio))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]

    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")

    extractor = get_ssl_extractor(args.ssl_model, device=args.device)

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
    )

    cb_config = args.codebook_config or f'{args.codebook_size}x{args.num_layers}'
    codebook_subdir = get_codebook_dir(args.output_dir, args.ssl_model, args.dataset, config=cb_config)
    output_path = codebook_subdir / f'ambiguity_{args.emotion}_{args.level}.pt'

    codebook_name = f"ambiguity_{args.emotion}_{args.level}"

    extra_metadata = {
        'experiment': 'ambiguity',
        'dataset': args.dataset,
        'emotion': args.emotion,
        'level': args.level,
        'target_consistency': target_consistency,
        'actual_consistency': actual_consistency,
        'composition': split_info.get('composition', {}),
        'num_samples': len(all_files),
    }

    best_loss = train_codebook(
        train_files=train_files,
        val_files=val_files,
        extractor=extractor,
        output_path=output_path,
        rvq_config=rvq_config,
        training_config=training_config,
        codebook_name=codebook_name,
        extra_metadata=extra_metadata,
        model_type=args.model_type,
    )

    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"  Best validation loss: {best_loss:.4f}")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  {args.emotion} / {args.level} (consistency={actual_consistency:.4f})")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
