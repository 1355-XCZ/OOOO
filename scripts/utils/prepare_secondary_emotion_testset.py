#!/usr/bin/env python3
"""
Prepare IEMOCAP Secondary-Emotion Test Set

Parses IEMOCAP evaluator annotations to find utterances where annotators
disagree, creating samples with primary + secondary emotion labels and
full vote distributions.

Two versions:
  va: ties double-counted (both orderings included), cap=100 per pair
  vb: ties excluded, no cap

Output:
  data/splits/iemocap/secondary_emotion_va.json
  data/splits/iemocap/secondary_emotion_vb.json

Usage:
  python scripts/utils/prepare_secondary_emotion_testset.py
  python scripts/utils/prepare_secondary_emotion_testset.py --version va
"""

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
TARGET_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

EMO_MAP = {
    'Anger': 'angry', 'ang': 'angry',
    'Happiness': 'happy', 'hap': 'happy',
    'Excited': 'happy', 'exc': 'happy',
    'Sadness': 'sad', 'sad': 'sad',
    'Neutral': 'neutral', 'neu': 'neutral',
}


def parse_iemocap_annotations(iemocap_root: Path):
    """Parse all EmoEvaluation files, return per-utterance vote lists."""
    annotation_files = sorted(iemocap_root.glob('**/EmoEvaluation/*.txt'))
    utt_votes = {}

    for ann_file in annotation_files:
        try:
            with open(ann_file, encoding='latin-1') as f:
                current_utt = None
                for line in f:
                    header = re.match(
                        r'\[[\d\.]+ - [\d\.]+\]\s+(\S+)\s+(\w+)\s+\[', line)
                    if header:
                        current_utt = header.group(1)
                        if current_utt not in utt_votes:
                            utt_votes[current_utt] = []
                        continue
                    indiv = re.match(r'C-[EF]\d:\s+(.+?);', line)
                    if indiv and current_utt:
                        raw_labels = [
                            s.strip() for s in indiv.group(1).split(';')
                            if s.strip()
                        ]
                        for raw in raw_labels:
                            mapped = EMO_MAP.get(raw)
                            if mapped and mapped in TARGET_EMOTIONS:
                                utt_votes[current_utt].append(mapped)
        except Exception:
            pass

    audio_files = {
        f.stem: str(f)
        for f in iemocap_root.glob('**/sentences/wav/**/*.wav')
    }

    return utt_votes, audio_files


def build_samples(utt_votes, audio_files):
    """Build candidate samples with vote distributions."""
    candidates = []
    for utt_id, votes in utt_votes.items():
        if utt_id not in audio_files:
            continue
        mapped = [v for v in votes if v in TARGET_EMOTIONS]
        if len(mapped) < 2:
            continue
        vote_counts = Counter(mapped)
        unique_emotions = [e for e in vote_counts if e in TARGET_EMOTIONS]
        if len(unique_emotions) < 2:
            continue

        most_common = vote_counts.most_common()
        primary_emo = most_common[0][0]
        primary_count = most_common[0][1]
        secondary_emo = most_common[1][0]
        secondary_count = most_common[1][1]
        is_tie = (primary_count == secondary_count)

        candidates.append({
            'utt_id': utt_id,
            'audio': audio_files[utt_id],
            'is_tie': is_tie,
            'votes': dict(vote_counts),
            'primary': primary_emo,
            'secondary': secondary_emo,
        })

    return candidates


def build_version_a(candidates, cap=100):
    """Version A: ties double-counted, cap per (primary, secondary) pair."""
    random.seed(SEED)
    pair_buckets = defaultdict(list)

    for s in candidates:
        pair_buckets[(s['primary'], s['secondary'])].append(s)
        if s['is_tie']:
            flipped = dict(s)
            flipped['primary'] = s['secondary']
            flipped['secondary'] = s['primary']
            pair_buckets[(flipped['primary'], flipped['secondary'])].append(flipped)

    samples = []
    pair_stats = {}
    for (pri, sec) in sorted(pair_buckets.keys()):
        bucket = pair_buckets[(pri, sec)]
        random.shuffle(bucket)
        selected = bucket[:cap]
        samples.extend(selected)
        pair_stats[f'{pri}->{sec}'] = len(selected)

    random.shuffle(samples)
    return samples, pair_stats


def build_version_b(candidates):
    """Version B: ties excluded, no cap."""
    random.seed(SEED)
    samples = [s for s in candidates if not s['is_tie']]
    pair_stats = Counter()
    for s in samples:
        pair_stats[f"{s['primary']}->{s['secondary']}"] += 1

    random.shuffle(samples)
    return samples, dict(sorted(pair_stats.items()))


def save_testset(samples, pair_stats, description, version):
    out_dir = SPLITS_DIR / 'iemocap'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'secondary_emotion_{version}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'description': description,
            'pair_stats': pair_stats,
            'samples': samples,
        }, f, indent=1)
    print(f'  Saved {len(samples)} samples -> {out_path}')
    print(f'  Pair stats: {pair_stats}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Prepare IEMOCAP secondary-emotion test set')
    parser.add_argument('--version', type=str, default='all',
                        choices=['va', 'vb', 'all'])
    args = parser.parse_args()

    iemocap_root = Path(DATA_ROOT) / 'IEMOCAP_full_release'
    if not iemocap_root.exists():
        print(f'ERROR: IEMOCAP not found at {iemocap_root}')
        sys.exit(1)

    print(f'Parsing IEMOCAP annotations from {iemocap_root} ...')
    utt_votes, audio_files = parse_iemocap_annotations(iemocap_root)
    print(f'  Found {len(utt_votes)} utterances, {len(audio_files)} audio files')

    candidates = build_samples(utt_votes, audio_files)
    print(f'  Candidates with secondary emotion: {len(candidates)}')

    versions = ['va', 'vb'] if args.version == 'all' else [args.version]

    for ver in versions:
        print(f'\n{"="*60}')
        print(f'  Building version: {ver}')
        print(f'{"="*60}')
        if ver == 'va':
            samples, pair_stats = build_version_a(candidates, cap=100)
            desc = 'Version A: ties double-counted, cap=100'
        else:
            samples, pair_stats = build_version_b(candidates)
            desc = 'Version B: ties excluded, no cap'
        save_testset(samples, pair_stats, desc, ver)

    print('\nDone.')


if __name__ == '__main__':
    main()
