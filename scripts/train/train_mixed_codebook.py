#!/usr/bin/env python3
"""
Train Mixed-Ratio Codebook

Trains a codebook using target-emotion data mixed with a small portion of other emotions.
E.g. 95% happy + 5% others (evenly distributed).

Args:
    --target-ratio: Target emotion proportion (0.0-1.0), e.g. 0.95 means 95% target + 5% others
    --dataset: Dataset name
    --emotion: Target emotion
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
    DEFAULT_LEARNING_RATE,
)
from core.config import set_seed, SSL_FEATURE_DIMS, SPLITS_DIR, CODEBOOK_DIR
from core.features import get_ssl_extractor, get_codebook_dir
from core.training import AudioFeatureDataset, collate_fn, train_codebook
from configs.dataset_config import DATASET_CONFIGS, RVQConfig, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mix_files(
    target_files: List[str],
    other_files_by_emotion: Dict[str, List[str]],
    target_ratio: float,
    seed: int = GLOBAL_SEED
) -> tuple:
    """Mix target-emotion files with other-emotion files at the given ratio.

    Returns:
        (mixed_files, mix_distribution) where mix_distribution = {emotion: count}.
    """
    rng = random.Random(seed)

    n_target = len(target_files)
    mix_distribution = {'target': n_target}

    if target_ratio >= 1.0 or not other_files_by_emotion:
        return list(target_files), mix_distribution

    # N_other = N_target * (1-ratio) / ratio
    n_other_total = int(n_target * (1.0 - target_ratio) / target_ratio)
    n_other_emotions = len(other_files_by_emotion)

    if n_other_emotions == 0 or n_other_total == 0:
        return list(target_files), mix_distribution

    # Distribute evenly across other emotions
    n_per_emotion = max(1, n_other_total // n_other_emotions)

    other_sampled = []
    for emo, files in other_files_by_emotion.items():
        n_sample = min(n_per_emotion, len(files))
        sampled = rng.sample(files, n_sample)
        other_sampled.extend(sampled)
        mix_distribution[emo] = n_sample

    mixed_files = list(target_files) + other_sampled
    rng.shuffle(mixed_files)

    return mixed_files, mix_distribution


def main():
    parser = argparse.ArgumentParser(description='Train mixed-ratio codebook for specific emotion')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (esd, iemocap, etc.)')
    parser.add_argument('--emotion', type=str, required=True,
                        help='Target emotion to train codebook')
    parser.add_argument('--target-ratio', type=float, required=True,
                        help='Target emotion ratio (e.g., 0.99, 0.95, 0.80)')
    parser.add_argument('--splits-dir', type=str,
                        default=str(SPLITS_DIR),
                        help='Directory containing split files')
    parser.add_argument('--ssl-model', type=str, default='e2v',
                        choices=['e2v', 'hubert', 'wavlm'],
                        help='SSL model for feature extraction')
    parser.add_argument('--output-dir', type=str,
                        default=str(CODEBOOK_DIR),
                        help='Output directory for codebooks')
    parser.add_argument('--codebook-config', type=str, default=None,
                        help='Codebook config tag (e.g. 2x32). Auto-derived from num-layers/codebook-size if omitted.')
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--num-layers', type=int, default=32, help='RVQ layers')
    parser.add_argument('--codebook-size', type=int, default=2, help='Codes per layer')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'encoder_decoder'],
                        help='Model architecture: standard (direct RVQ) or encoder_decoder (encoder-RVQ-decoder)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset_config = DATASET_CONFIGS[args.dataset]
    if not dataset_config.enabled:
        raise ValueError(f"Dataset {args.dataset} is disabled")

    # Validate emotion
    if args.emotion not in dataset_config.emotions:
        raise ValueError(f"Emotion {args.emotion} not in {args.dataset}. Available: {dataset_config.emotions}")

    # Validate ratio
    if not (0.0 < args.target_ratio <= 1.0):
        raise ValueError(f"target-ratio must be in (0, 1], got {args.target_ratio}")

    ratio_int = int(args.target_ratio * 100)
    feature_dim = SSL_FEATURE_DIMS[args.ssl_model]

    logger.info("=" * 60)
    logger.info(f"Training Mixed-Ratio Codebook")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Emotion: {args.emotion}")
    logger.info(f"  SSL Model: {args.ssl_model} (dim={feature_dim})")
    logger.info(f"  Target Ratio: {args.target_ratio} ({ratio_int}%)")
    logger.info("=" * 60)

    # Load split files
    splits_dir = Path(args.splits_dir) / args.dataset

    with open(splits_dir / 'train.json') as f:
        train_splits = json.load(f)
    with open(splits_dir / 'val.json') as f:
        val_splits = json.load(f)

    # Get target emotion files
    target_train_files = train_splits.get(args.emotion, [])
    target_val_files = val_splits.get(args.emotion, [])

    if not target_train_files:
        raise ValueError(f"No training files found for emotion {args.emotion}")

    # Get other-emotion files
    other_emotions = [e for e in dataset_config.emotions if e != args.emotion]
    other_train_by_emo = {e: train_splits.get(e, []) for e in other_emotions if train_splits.get(e)}
    other_val_by_emo = {e: val_splits.get(e, []) for e in other_emotions if val_splits.get(e)}

    logger.info(f"Target emotion '{args.emotion}': {len(target_train_files)} train, {len(target_val_files)} val")
    for emo, files in other_train_by_emo.items():
        logger.info(f"  Other emotion '{emo}': {len(files)} train available")

    # Mix files
    train_files, train_mix_dist = mix_files(
        target_train_files, other_train_by_emo, args.target_ratio, seed=args.seed
    )
    val_files, val_mix_dist = mix_files(
        target_val_files, other_val_by_emo, args.target_ratio, seed=args.seed + 1
    )

    logger.info(f"\nMixed training set: {len(train_files)} files")
    logger.info(f"  Distribution: {train_mix_dist}")
    logger.info(f"Mixed validation set: {len(val_files)} files")
    logger.info(f"  Distribution: {val_mix_dist}")

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
    output_path = codebook_subdir / f'mixed_{args.emotion}_r{ratio_int}.pt'

    codebook_name = f"mixed_{args.emotion}_r{ratio_int}"

    # Extra metadata
    extra_metadata = {
        'target_ratio': args.target_ratio,
        'target_emotion': args.emotion,
        'mix_distribution': {
            'train': train_mix_dist,
            'val': val_mix_dist,
        },
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
    logger.info(f"  Target ratio: {args.target_ratio} ({ratio_int}%)")
    logger.info(f"  Train mix: {train_mix_dist}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
