#!/usr/bin/env python3
"""
Train Balanced Codebook

Balanced codebook: equal sampling from each emotion, total training data matches the biased codebook.
"""

import json
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import (
    GLOBAL_SEED, DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_PATIENCE, DEFAULT_MIN_DELTA,
)

import torch
import numpy as np

from core.config import set_seed, SSL_FEATURE_DIMS, CODEBOOK_DIR, SPLITS_DIR
from core.features import get_ssl_extractor, get_codebook_dir
from core.training import AudioFeatureDataset, collate_fn, train_codebook
from configs.dataset_config import DATASET_CONFIGS, RVQConfig, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_balanced_dataset(
    files_by_emotion: Dict[str, List[str]],
    total_samples: int,
    seed: int = GLOBAL_SEED
) -> List[str]:
    """Create a balanced dataset: equal sampling from each emotion."""
    random.seed(seed)

    emotions = list(files_by_emotion.keys())
    n_emotions = len(emotions)
    samples_per_emotion = total_samples // n_emotions

    logger.info(f"Creating balanced dataset: {samples_per_emotion}/emotion x {n_emotions} = {total_samples}")

    balanced_files = []
    for emotion in emotions:
        files = files_by_emotion[emotion]
        if len(files) >= samples_per_emotion:
            sampled = random.sample(files, samples_per_emotion)
        else:
            logger.warning(f"  {emotion}: only {len(files)} files, padding with repeats")
            sampled = files.copy()
            while len(sampled) < samples_per_emotion:
                sampled.extend(random.sample(files, min(len(files), samples_per_emotion - len(sampled))))
        balanced_files.extend(sampled)
        logger.info(f"  {emotion}: {len(sampled)} files")

    random.shuffle(balanced_files)
    return balanced_files


def main():
    parser = argparse.ArgumentParser(description='Train balanced codebook with equal samples from all emotions')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--ssl-model', type=str, default='e2v',
                        choices=['e2v', 'hubert', 'wavlm'])
    parser.add_argument('--splits-dir', type=str, default=str(SPLITS_DIR))
    parser.add_argument('--output-dir', type=str, default=str(CODEBOOK_DIR))
    parser.add_argument('--codebook-config', type=str, default=None,
                        help='Codebook config tag (e.g. 2x32). Auto-derived from num-layers/codebook-size if omitted.')
    parser.add_argument('--samples-per-emotion', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--num-layers', type=int, default=32, help='RVQ layers')
    parser.add_argument('--codebook-size', type=int, default=2, help='Codes per layer')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'encoder_decoder'])
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=DEFAULT_MIN_DELTA,
                        help='Minimum loss improvement to reset patience counter')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    parser.add_argument('--codebook-name', type=str, default='balanced',
                        help='Name for the codebook file (default: balanced). '
                             'Use "balanced_fair" for data-budget-matched variant.')
    parser.add_argument('--test', action='store_true',
                        help='Smoke-test mode: tiny data, few epochs')
    args = parser.parse_args()

    if args.test:
        print("=" * 60)
        print("  TEST MODE -- smoke-test with minimal data/epochs")
        print("=" * 60)
        args.num_epochs = min(args.num_epochs, 2)
        args.batch_size = min(args.batch_size, 4)
        args.codebook_size = min(args.codebook_size, 4)
        args.num_layers = min(args.num_layers, 2)
        if args.samples_per_emotion is None or args.samples_per_emotion > 5:
            args.samples_per_emotion = 5

    set_seed(args.seed)

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    dataset_config = DATASET_CONFIGS[args.dataset]
    if not dataset_config.enabled:
        raise ValueError(f"Dataset {args.dataset} is disabled")

    feature_dim = SSL_FEATURE_DIMS[args.ssl_model]

    logger.info("=" * 60)
    logger.info(f"Training Balanced Codebook")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  SSL Model: {args.ssl_model} (dim={feature_dim})")
    logger.info("=" * 60)

    # Load splits
    splits_dir = Path(args.splits_dir) / args.dataset
    with open(splits_dir / 'train.json') as f:
        train_splits = json.load(f)
    with open(splits_dir / 'val.json') as f:
        val_splits = json.load(f)

    logger.info("Emotion distribution in training set:")
    min_samples = float('inf')
    for emotion, files in train_splits.items():
        logger.info(f"  {emotion}: {len(files)}")
        min_samples = min(min_samples, len(files))

    n_emotions = len(train_splits)

    if args.samples_per_emotion:
        samples_per_emotion = args.samples_per_emotion
        total_samples = samples_per_emotion * n_emotions
    else:
        total_samples = min_samples
        samples_per_emotion = total_samples // n_emotions

    logger.info(f"Balanced: {samples_per_emotion}/emotion x {n_emotions} = {total_samples} total")

    train_files = create_balanced_dataset(train_splits, total_samples, args.seed)
    val_files = create_balanced_dataset(
        val_splits, len(val_splits) * samples_per_emotion // len(train_splits), args.seed + 1
    )

    logger.info(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

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
        patience=args.patience,
        min_delta=args.min_delta,
    )

    cb_config = args.codebook_config or f'{args.codebook_size}x{args.num_layers}'
    codebook_subdir = get_codebook_dir(args.output_dir, args.ssl_model, args.dataset, config=cb_config)
    output_path = codebook_subdir / f'{args.codebook_name}.pt'

    best_loss = train_codebook(
        train_files=train_files,
        val_files=val_files,
        extractor=extractor,
        output_path=output_path,
        rvq_config=rvq_config,
        training_config=training_config,
        codebook_name=args.codebook_name,
        model_type=args.model_type,
    )

    logger.info("=" * 60)
    logger.info(f"Training complete! Best loss: {best_loss:.4f}")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
