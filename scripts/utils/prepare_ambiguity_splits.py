#!/usr/bin/env python3
"""
Generate Ambiguity Splits for IEMOCAP and CREMA-D

Both datasets produce splits with identical structure:
  {emotion}_{level}: { files: [...], target_consistency, actual_consistency, composition, num_samples }

Consistency levels:
  high = 100%  (all annotators agree)
  mid  ~ 80%   (mixed to achieve ~0.80 average)
  low  ~ 67%   (mixed to achieve ~0.67 average)

Sample count per codebook: 152 (controlled for fair comparison)

Usage:
  python prepare_ambiguity_splits.py --dataset iemocap
  python prepare_ambiguity_splits.py --dataset cremad
  python prepare_ambiguity_splits.py --dataset all
"""

import csv
import json
import re
import random
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import GLOBAL_SEED, DATA_ROOT
from core.config import SPLITS_DIR

SEED = GLOBAL_SEED
SAMPLES_PER_CODEBOOK = 152
TARGET_CONSISTENCY = {'high': 1.0, 'mid': 0.80, 'low': 0.67}
AMBIGUITY_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']


# ============================================================
# IEMOCAP: parse individual annotator votes
# ============================================================

def load_iemocap_samples(iemocap_root: Path):
    """Load IEMOCAP samples with per-utterance agreement scores."""
    annotation_files = list(iemocap_root.glob('**/EmoEvaluation/*.txt'))
    utterance_annotations = {}
    utterance_consensus = {}

    for ann_file in annotation_files:
        try:
            with open(ann_file, encoding='latin-1') as f:
                current_utt = None
                for line in f:
                    match = re.match(r'\[[\d\.]+ - [\d\.]+\]\s+(\S+)\s+(\w+)\s+\[', line)
                    if match:
                        current_utt = match.group(1)
                        utterance_consensus[current_utt] = match.group(2)
                        if current_utt not in utterance_annotations:
                            utterance_annotations[current_utt] = []
                        continue
                    indiv_match = re.match(r'C-[EF]\d:\s+(\w+)', line)
                    if indiv_match and current_utt:
                        utterance_annotations[current_utt].append(indiv_match.group(1))
        except Exception:
            pass

    audio_files = {f.stem: str(f) for f in iemocap_root.glob('**/sentences/wav/**/*.wav')}

    emo_map = {
        'Anger': 'angry', 'ang': 'angry',
        'Happiness': 'happy', 'hap': 'happy',
        'Excited': 'happy', 'exc': 'happy',
        'Sadness': 'sad', 'sad': 'sad',
        'Neutral': 'neutral', 'neu': 'neutral',
    }

    samples_by_emotion = defaultdict(list)
    for utt_id, votes in utterance_annotations.items():
        if len(votes) < 2 or utt_id not in audio_files:
            continue
        mapped = [emo_map.get(v) for v in votes]
        mapped = [m for m in mapped if m is not None]
        if not mapped:
            continue
        vote_counts = Counter(mapped)
        majority_emo, majority_count = vote_counts.most_common(1)[0]
        if majority_emo not in AMBIGUITY_EMOTIONS:
            continue
        agreement = majority_count / len(mapped)
        samples_by_emotion[majority_emo].append({
            'file': audio_files[utt_id],
            'agreement': round(agreement, 4),
        })

    return dict(samples_by_emotion)


# ============================================================
# CREMA-D: parse crowd-sourced tabulatedVotes.csv
# ============================================================

def load_cremad_samples(cremad_root: Path):
    """Load CREMA-D samples with per-utterance agreement from crowd votes."""
    votes_path = cremad_root / 'processedResults' / 'tabulatedVotes.csv'
    if not votes_path.exists():
        raise FileNotFoundError(
            f"CREMA-D voting data not found: {votes_path}\n"
            "Download from: https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/"
            "master/processedResults/tabulatedVotes.csv"
        )

    audio_dir = None
    for d in [cremad_root / 'versions' / '1' / 'AudioWAV', cremad_root / 'AudioWAV', cremad_root]:
        if d.exists() and list(d.glob('*.wav'))[:1]:
            audio_dir = d
            break
    if audio_dir is None:
        raise FileNotFoundError(f"CREMA-D audio not found under {cremad_root}")

    audio_map = {f.stem: str(f) for f in audio_dir.glob('*.wav')}
    intended_emo_code = {}
    for stem in audio_map:
        parts = stem.split('_')
        if len(parts) >= 3:
            intended_emo_code[stem] = parts[2]

    code_to_emo = {'ANG': 'angry', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}

    samples_by_emotion = defaultdict(list)
    with open(votes_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['fileName'].strip()
            if fname not in audio_map:
                continue
            emo_code = intended_emo_code.get(fname)
            if emo_code not in code_to_emo:
                continue
            emo = code_to_emo[emo_code]
            agreement = float(row['agreement'])
            samples_by_emotion[emo].append({
                'file': audio_map[fname],
                'agreement': round(agreement, 4),
            })

    return dict(samples_by_emotion)


# ============================================================
# Sample selection: mix to achieve target average consistency
# ============================================================

def select_samples(samples, target, n):
    """Select n samples whose average agreement ~ target."""
    random.seed(SEED)

    if target >= 1.0:
        perfect = [s for s in samples if s['agreement'] >= 1.0]
        random.shuffle(perfect)
        selected = perfect[:n]
        if len(selected) < n:
            print(f"  WARNING: only {len(selected)}/{n} samples with 100% agreement")
        return selected

    high_pool = sorted([s for s in samples if s['agreement'] >= target],
                       key=lambda s: s['agreement'], reverse=True)
    low_pool = sorted([s for s in samples if s['agreement'] < target],
                      key=lambda s: s['agreement'], reverse=True)
    random.shuffle(high_pool)
    random.shuffle(low_pool)

    selected = []
    current_sum = 0.0
    hi, lo = 0, 0

    while len(selected) < n and (hi < len(high_pool) or lo < len(low_pool)):
        current_avg = current_sum / len(selected) if selected else 0
        if current_avg < target and hi < len(high_pool):
            s = high_pool[hi]; hi += 1
        elif current_avg > target and lo < len(low_pool):
            s = low_pool[lo]; lo += 1
        elif hi < len(high_pool):
            s = high_pool[hi]; hi += 1
        elif lo < len(low_pool):
            s = low_pool[lo]; lo += 1
        else:
            break
        selected.append(s)
        current_sum += s['agreement']

    return selected


# ============================================================
# Build splits dict
# ============================================================

def build_splits(samples_by_emotion, dataset_name):
    """Build ambiguity splits for all emotions x levels."""
    splits = {}
    print(f"\n{'='*60}")
    print(f"  {dataset_name} Ambiguity Splits (n={SAMPLES_PER_CODEBOOK})")
    print(f"{'='*60}")

    for emo in AMBIGUITY_EMOTIONS:
        pool = samples_by_emotion.get(emo, [])
        if not pool:
            print(f"  {emo}: NO SAMPLES -- skipping")
            continue

        for level, target in TARGET_CONSISTENCY.items():
            key = f"{emo}_{level}"
            selected = select_samples(pool, target, SAMPLES_PER_CODEBOOK)

            agreements = [s['agreement'] for s in selected]
            actual = sum(agreements) / len(agreements) if agreements else 0.0
            composition = dict(Counter(round(a, 4) for a in agreements).most_common())

            splits[key] = {
                'files': [s['file'] for s in selected],
                'target_consistency': target,
                'actual_consistency': round(actual, 4),
                'composition': {str(k): v for k, v in sorted(composition.items(), reverse=True)},
                'num_samples': len(selected),
            }

            status = "OK" if len(selected) == SAMPLES_PER_CODEBOOK else f"SHORT ({len(selected)})"
            print(f"  {key:<16}: n={len(selected):>3}, target={target:.0%}, "
                  f"actual={actual:.4f}, [{status}]")

    return splits


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate ambiguity splits')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['iemocap', 'cremad', 'all'])
    parser.add_argument('--iemocap-root', type=str,
                        default=f'{DATA_ROOT}/IEMOCAP_full_release')
    parser.add_argument('--cremad-root', type=str,
                        default=f'{DATA_ROOT}/CREMA-D')
    parser.add_argument('--output-dir', type=str, default=str(SPLITS_DIR))
    args = parser.parse_args()

    random.seed(SEED)

    datasets_to_process = []
    if args.dataset in ('iemocap', 'all'):
        datasets_to_process.append(('iemocap', args.iemocap_root))
    if args.dataset in ('cremad', 'all'):
        datasets_to_process.append(('cremad', args.cremad_root))

    for ds_name, ds_root in datasets_to_process:
        ds_root = Path(ds_root)
        print(f"\nLoading {ds_name} samples from {ds_root}...")

        if ds_name == 'iemocap':
            samples = load_iemocap_samples(ds_root)
        else:
            samples = load_cremad_samples(ds_root)

        for emo in AMBIGUITY_EMOTIONS:
            pool = samples.get(emo, [])
            agr_vals = [s['agreement'] for s in pool]
            n100 = sum(1 for a in agr_vals if a >= 1.0)
            print(f"  {emo}: total={len(pool)}, 100%={n100}")

        splits = build_splits(samples, ds_name.upper())

        out_dir = Path(args.output_dir) / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'ambiguity_splits.json'
        with open(out_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"\n  Saved to: {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
