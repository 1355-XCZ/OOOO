#!/usr/bin/env python3
"""
Train SER Linear Probe Classifiers

For each (SSL model, dataset) pair, trains a simple linear classifier on
raw SSL features (mean pooling + Linear) using existing train/val splits.

Usage:
    python train_ser_classifier.py --ssl-model e2v --dataset esd
    python train_ser_classifier.py --ssl-model hubert --dataset iemocap
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import GLOBAL_SEED, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_PATIENCE
from core.config import set_seed, SSL_FEATURE_DIMS, SPLITS_DIR, CLASSIFIER_DIR
from core.features import get_ssl_extractor, extract_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Feature caching (extract_features from core.features)
# ============================================================

def pre_extract_features(
    extractor,
    samples: List[Tuple[str, int]],
) -> List[Tuple[torch.Tensor, int]]:
    """
    Pre-extract SSL features for all samples and cache in memory.

    Args:
        extractor: SSL feature extractor
        samples: list of (audio_path, label_idx)

    Returns:
        list of (features_tensor, label_idx) for successfully extracted samples
    """
    cached = []
    for audio_path, label_idx in tqdm(samples, desc="Pre-extracting features"):
        feats = extract_features(extractor, audio_path)
        if feats is not None and feats.dim() == 2 and feats.size(0) > 0:
            cached.append((feats, label_idx))
    logger.info(f"Cached {len(cached)}/{len(samples)} samples")
    return cached


# ============================================================
# Dataset for cached features
# ============================================================

class CachedFeatureDataset(Dataset):
    """Dataset that serves pre-extracted, in-memory features."""

    def __init__(self, cached_samples: List[Tuple[torch.Tensor, int]]):
        self.samples = cached_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        return feats, feats.size(0), label


def collate_fn(batch):
    """Collate variable-length feature sequences with padding."""
    feats_list, lengths, labels = zip(*batch)
    max_len = max(lengths)
    feature_dim = feats_list[0].size(-1)

    padded = torch.zeros(len(feats_list), max_len, feature_dim)
    for i, (f, l) in enumerate(zip(feats_list, lengths)):
        padded[i, :l] = f

    lengths = torch.LongTensor(lengths)
    labels = torch.LongTensor(labels)
    return padded, lengths, labels


# ============================================================
# Linear Probe model
# ============================================================

class LinearProbe(nn.Module):
    """Simple mean-pooling + linear classifier."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor, features_len: torch.Tensor):
        # Masked mean pooling
        mask = torch.arange(features.size(1), device=features.device)[None, :]
        mask = (mask < features_len[:, None]).unsqueeze(-1).float()
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# ============================================================
# Training & evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for features, lengths, labels in loader:
        features = features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features, lengths)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for features, lengths, labels in loader:
        features = features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(features, lengths)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train SER Linear Probe Classifier")
    parser.add_argument('--ssl-model', type=str, required=True,
                        choices=['e2v', 'hubert', 'wavlm'],
                        help='SSL model for feature extraction')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['esd', 'esd_en', 'iemocap', 'ravdess', 'cremad', 'emodb', 'msp'],
                        help='Dataset to train on')
    parser.add_argument('--splits-dir', type=str,
                        default=str(SPLITS_DIR),
                        help='Directory containing data splits')
    parser.add_argument('--output-dir', type=str,
                        default=str(CLASSIFIER_DIR),
                        help='Base output directory for classifiers')
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    feature_dim = SSL_FEATURE_DIMS[args.ssl_model]

    # ---- Output directory ----
    out_dir = Path(args.output_dir) / args.ssl_model / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Train SER Linear Probe Classifier")
    logger.info(f"  SSL Model : {args.ssl_model} (dim={feature_dim})")
    logger.info(f"  Dataset   : {args.dataset}")
    logger.info(f"  Output    : {out_dir}")
    logger.info(f"  Epochs    : {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR        : {args.lr}")
    logger.info(f"  Patience  : {args.patience}")
    logger.info(f"  Device    : {device}")
    logger.info("=" * 60)

    # ---- Load data splits ----
    splits_dir = Path(args.splits_dir) / args.dataset
    with open(splits_dir / 'train.json') as f:
        train_splits = json.load(f)
    with open(splits_dir / 'val.json') as f:
        val_splits = json.load(f)

    # Build emotion label mapping (sorted for reproducibility)
    emotions = sorted(train_splits.keys())
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}
    num_classes = len(emotions)

    logger.info(f"Emotions ({num_classes}): {emotions}")

    # Flatten into (audio_path, label_idx) lists
    train_samples = []
    for emo, files in train_splits.items():
        idx = emotion_to_idx[emo]
        for f in files:
            train_samples.append((f, idx))

    val_samples = []
    for emo, files in val_splits.items():
        if emo in emotion_to_idx:
            idx = emotion_to_idx[emo]
            for f in files:
                val_samples.append((f, idx))

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples  : {len(val_samples)}")

    # ---- Pre-extract SSL features ----
    logger.info(f"\nLoading {args.ssl_model} extractor ...")
    extractor = get_ssl_extractor(args.ssl_model, device=device)

    logger.info("Pre-extracting training features ...")
    cached_train = pre_extract_features(extractor, train_samples)
    logger.info("Pre-extracting validation features ...")
    cached_val = pre_extract_features(extractor, val_samples)

    # Free extractor to save GPU memory
    del extractor
    torch.cuda.empty_cache()

    logger.info(f"Usable train: {len(cached_train)}, val: {len(cached_val)}")

    # ---- Create dataloaders ----
    train_dataset = CachedFeatureDataset(cached_train)
    val_dataset = CachedFeatureDataset(cached_val)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # ---- Create model ----
    model = LinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Model: {model}")

    # ---- Training loop ----
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:3d}/{args.num_epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'emotions': emotions,
                'emotion_to_idx': emotion_to_idx,
                'feature_dim': feature_dim,
                'num_classes': num_classes,
                'ssl_model': args.ssl_model,
                'dataset': args.dataset,
                'epoch': epoch,
                'val_acc': float(val_acc),
                'val_f1': float(val_f1),
            }
            torch.save(checkpoint, out_dir / 'best_model.pt')
            logger.info(f"  => Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"  Best epoch    : {best_epoch}")
    logger.info(f"  Best val acc  : {best_val_acc:.4f}")
    logger.info(f"  Saved to      : {out_dir / 'best_model.pt'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
