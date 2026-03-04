#!/usr/bin/env python3
"""
Prepare IEMOCAP Secondary-Emotion Test Set

Parses IEMOCAP multi-annotator voting data to extract samples with both
primary and secondary emotions from the fair-4 set {angry, happy, neutral, sad}.

Note: 'excited' (exc) and 'happiness' (hap) are treated as the same emotion ('happy').

Produces two versions:
  Version A (main):   Tied samples are counted in BOTH directions.
  Version B (strict): Tied samples are excluded entirely.

Output:
  data/splits/iemocap/secondary_emotion_v{a,b}.json

Each entry: {utt_id, audio, primary, secondary, is_tie, votes}
"""

import re
import json
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import GLOBAL_SEED, DATA_ROOT
from core.config import SPLITS_DIR

IEMOCAP_ROOT = Path(DATA_ROOT) / 'IEMOCAP_full_release'
FAIR4 = ['angry', 'happy', 'neutral', 'sad']
MAX_PER_PAIR = 100

EMO_MAP = {
    'Anger': 'angry', 'ang': 'angry',
    'Happiness': 'happy', 'hap': 'happy',
    'Excited': 'happy', 'exc': 'happy',
    'Sadness': 'sad', 'sad': 'sad',
    'Neutral': 'neutral', 'neu': 'neutral',
}


def parse_iemocap_votes(iemocap_root: Path):
    """Parse individual evaluator votes from IEMOCAP annotation files."""
    annotation_files = list(iemocap_root.glob('**/EmoEvaluation/*.txt'))
    utterance_annotations = {}
    utterance_consensus = {}

    for ann_file in annotation_files:
        try:
            with open(ann_file, encoding='latin-1') as f:
                current_utt = None
                for line in f:
                    match = re.match(
                        r'\[[\d\.]+ - [\d\.]+\]\s+(\S+)\s+(\w+)\s+\[', line)
                    if match:
                        current_utt = match.group(1)
                        utterance_consensus[current_utt] = match.group(2)
                        if current_utt not in utterance_annotations:
                            utterance_annotations[current_utt] = []
                        continue
                    indiv_match = re.match(r'C-[EF]\d:\s+(\w+)', line)
                    if indiv_match and current_utt:
                        utterance_annotations[current_utt].append(
                            indiv_match.group(1))
        except Exception:
            pass

    audio_files = {
        f.stem: str(f)
        for f in iemocap_root.glob('**/sentences/wav/**/*.wav')
    }

    return utterance_annotations, utterance_consensus, audio_files


def build_testsets(utterance_annotations, utterance_consensus, audio_files,
                   seed=GLOBAL_SEED):
    """Build Version A and Version B test sets."""
    rng = random.Random(seed)
    fair4_set = set(FAIR4)

    raw_parsed = []
    for utt_id, raw_votes in utterance_annotations.items():
        if utt_id not in audio_files:
            continue
        mapped = [EMO_MAP.get(v) for v in raw_votes]
        mapped = [m for m in mapped if m is not None]
        if len(mapped) < 2:
            continue
        vote_counts = Counter(mapped)
        ranked = vote_counts.most_common()
        if len(ranked) < 2:
            continue
        if ranked[0][0] not in fair4_set or ranked[1][0] not in fair4_set:
            continue
        is_tie = ranked[0][1] == ranked[1][1]
        tied_emotions = [e for e, c in ranked if c == ranked[0][1] and e in fair4_set]
        raw_parsed.append({
            'utt_id': utt_id,
            'emo1': ranked[0][0],
            'emo2': ranked[1][0],
            'tied_emotions': tied_emotions,
            'is_tie': is_tie,
            'audio': audio_files[utt_id],
            'votes': dict(vote_counts),
        })

    # Version A: ties counted in all directions
    pair_a = defaultdict(list)
    for s in raw_parsed:
        entry = {
            'utt_id': s['utt_id'],
            'audio': s['audio'],
            'is_tie': s['is_tie'],
            'votes': s['votes'],
        }
        if s['is_tie']:
            tied = s['tied_emotions']
            for i, emo_p in enumerate(tied):
                for j, emo_s in enumerate(tied):
                    if i != j:
                        pair_a[(emo_p, emo_s)].append(
                            {**entry, 'primary': emo_p, 'secondary': emo_s})
        else:
            pair_a[(s['emo1'], s['emo2'])].append(
                {**entry, 'primary': s['emo1'], 'secondary': s['emo2']})

    # Version B: exclude all ties
    pair_b = defaultdict(list)
    for s in raw_parsed:
        if s['is_tie']:
            continue
        pair_b[(s['emo1'], s['emo2'])].append({
            'utt_id': s['utt_id'],
            'audio': s['audio'],
            'primary': s['emo1'],
            'secondary': s['emo2'],
            'is_tie': False,
            'votes': s['votes'],
        })

    def _cap_and_flatten(pair_dict, max_per_pair):
        samples = []
        pair_stats = {}
        for primary in FAIR4:
            for secondary in FAIR4:
                if primary == secondary:
                    continue
                entries = pair_dict.get((primary, secondary), [])
                if len(entries) > max_per_pair:
                    entries = rng.sample(entries, max_per_pair)
                pair_stats[f'{primary}->{secondary}'] = len(entries)
                samples.extend(entries)
        return samples, pair_stats

    va_samples, va_stats = _cap_and_flatten(pair_a, MAX_PER_PAIR)
    vb_samples, vb_stats = _cap_and_flatten(pair_b, max_per_pair=9999)

    return va_samples, va_stats, vb_samples, vb_stats


def main():
    print(f'Parsing IEMOCAP from {IEMOCAP_ROOT}')
    annotations, consensus, audio = parse_iemocap_votes(IEMOCAP_ROOT)
    print(f'  {len(annotations)} utterances, {len(audio)} audio files')

    va, va_stats, vb, vb_stats = build_testsets(annotations, consensus, audio)

    out_dir = SPLITS_DIR / 'iemocap'
    out_dir.mkdir(parents=True, exist_ok=True)

    va_path = out_dir / 'secondary_emotion_va.json'
    with open(va_path, 'w') as f:
        json.dump({'samples': va, 'pair_stats': va_stats,
                   'description': 'Version A: ties double-counted, cap=100'},
                  f, indent=1)
    print(f'\nVersion A saved: {va_path}')
    print(f'  {len(va)} samples')
    for k, v in sorted(va_stats.items()):
        print(f'    {k}: {v}')

    vb_path = out_dir / 'secondary_emotion_vb.json'
    with open(vb_path, 'w') as f:
        json.dump({'samples': vb, 'pair_stats': vb_stats,
                   'description': 'Version B: ties excluded, no cap'},
                  f, indent=1)
    print(f'\nVersion B saved: {vb_path}')
    print(f'  {len(vb)} samples')
    for k, v in sorted(vb_stats.items()):
        print(f'    {k}: {v}')


if __name__ == '__main__':
    main()
