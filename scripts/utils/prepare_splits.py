#!/usr/bin/env python3
"""
Prepare Train/Val/Test Splits for All Datasets

Default split ratios: train 50% / val 10% / test 40%.
Stratified by emotion to maintain consistent proportions across splits.
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from configs.constants import GLOBAL_SEED, COLLAB_DATA_ROOT
from configs.dataset_config import (
    DATASET_CONFIGS, get_enabled_datasets, 
    TrainingConfig, DEFAULT_TRAINING_CONFIG,
    CAMEO_DATASETS,
)


def load_esd_files(config, speaker_filter=None) -> Dict[str, List[str]]:
    """Load ESD dataset files grouped by emotion.
    
    Args:
        config: DatasetConfig
        speaker_filter: optional set/list of speaker IDs (int) to include.
                        e.g. range(11, 21) for English-only speakers.
    """
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    for emotion in config.emotions:
        emotion_dir = emotion.capitalize() if emotion != 'surprise' else 'Surprise'
        for speaker_dir in data_root.iterdir():
            if not speaker_dir.is_dir():
                continue
            if speaker_filter is not None:
                try:
                    spk_id = int(speaker_dir.name)
                    if spk_id not in speaker_filter:
                        continue
                except ValueError:
                    continue
            emotion_path = speaker_dir / emotion_dir
            if emotion_path.exists():
                wav_files = list(emotion_path.glob('*.wav'))
                files_by_emotion[emotion].extend([str(f) for f in wav_files])
    
    return dict(files_by_emotion)


def load_iemocap_files(config) -> Dict[str, List[str]]:
    """Load IEMOCAP dataset files grouped by emotion."""
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    annotation_files = list(data_root.glob('**/EmoEvaluation/*.txt'))
    
    utt_to_emotion = {}
    for ann_file in annotation_files:
        try:
            with open(ann_file, encoding='latin-1') as f:
                for line in f:
                    match = re.match(r'\[[\d\.]+ - [\d\.]+\]\s+(\S+)\s+(\w+)\s+\[', line)
                    if match:
                        utt_id = match.group(1)
                        emotion_raw = match.group(2)
                        
                        emotion_map = {
                            'ang': 'angry',
                            'hap': 'happy',
                            'exc': 'happy',  # merge excited into happy
                            'sad': 'sad',
                            'neu': 'neutral',
                            'fru': None,  # skip frustrated
                            'sur': None,  # skip surprise (too few)
                            'fea': None,  # skip fear (too few)
                            'dis': None,  # skip disgust (too few)
                            'xxx': None,
                            'oth': None,
                        }
                        
                        if emotion_raw in emotion_map and emotion_map[emotion_raw]:
                            utt_to_emotion[utt_id] = emotion_map[emotion_raw]
        except Exception as e:
            print(f"Warning: Failed to parse {ann_file}: {e}")
    
    # Find audio files
    wav_files = list(data_root.glob('**/*.wav'))
    for wav_file in wav_files:
        utt_id = wav_file.stem
        if utt_id in utt_to_emotion:
            emotion = utt_to_emotion[utt_id]
            files_by_emotion[emotion].append(str(wav_file))
    
    return dict(files_by_emotion)


def load_ravdess_files(config) -> Dict[str, List[str]]:
    """Load RAVDESS dataset files grouped by emotion."""
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    emotion_code_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    wav_files = list(data_root.glob('**/*.wav'))
    for wav_file in wav_files:
        # Filename format: XX-XX-XX-XX-XX-XX-XX.wav; 3rd field is emotion code
        parts = wav_file.stem.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_code_map:
                emotion = emotion_code_map[emotion_code]
                if emotion in config.emotions:
                    files_by_emotion[emotion].append(str(wav_file))
    
    return dict(files_by_emotion)


def load_cremad_files(config) -> Dict[str, List[str]]:
    """Load CREMA-D dataset files grouped by emotion."""
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    emotion_code_map = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
        'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
    }
    
    # CREMA-D directory: data_root/versions/1/AudioWAV/*.wav
    possible_dirs = [
        data_root / 'versions' / '1' / 'AudioWAV',
        data_root / 'AudioWAV',
        data_root
    ]
    
    audio_dir = None
    for d in possible_dirs:
        if d.exists():
            wav_files = list(d.glob('*.wav'))
            if wav_files:
                audio_dir = d
                break
    
    if audio_dir is None:
        print(f"Warning: CREMA-D audio directory not found in {data_root}")
        return dict(files_by_emotion)
    
    wav_files = list(audio_dir.glob('*.wav'))
    for wav_file in wav_files:
        # Filename format: 1001_DFA_ANG_XX.wav
        parts = wav_file.stem.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_code_map:
                emotion = emotion_code_map[emotion_code]
                if emotion in config.emotions:
                    files_by_emotion[emotion].append(str(wav_file))
    
    return dict(files_by_emotion)


def load_emodb_files(config) -> Dict[str, List[str]]:
    """Load EmoDB dataset files grouped by emotion."""
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    emotion_code_map = {
        'W': 'anger', 'L': 'boredom', 'E': 'disgust',
        'A': 'fear', 'F': 'happiness', 'N': 'neutral', 'T': 'sadness'
    }
    
    # EmoDB may store wavs in a wav/ subdirectory
    search_dirs = [data_root, data_root / 'wav']
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        wav_files = list(search_dir.glob('*.wav'))
        for wav_file in wav_files:
            # e.g. 03a01Wa.wav -> W is the emotion code
            name = wav_file.stem
            emotion_code = None
            for i in range(len(name) - 1, -1, -1):
                if name[i].isalpha() and name[i].upper() in emotion_code_map:
                    emotion_code = name[i].upper()
                    break
            
            if emotion_code and emotion_code in emotion_code_map:
                emotion = emotion_code_map[emotion_code]
                if emotion in config.emotions:
                    files_by_emotion[emotion].append(str(wav_file))
    
    return dict(files_by_emotion)


def load_msp_files(config) -> Dict[str, List[str]]:
    """Load MSP-Podcast dataset files grouped by emotion."""
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    json_path = Path(COLLAB_DATA_ROOT) / 'msp_ambigous.json'
    
    if not json_path.exists():
        print(f"Warning: MSP JSON file not found: {json_path}")
        return dict(files_by_emotion)
    
    with open(json_path) as f:
        msp_data = json.load(f)
    
    for item in msp_data:
        # emotion field is a list; use majority vote
        emotions_list = item.get('emotion', [])
        audio_path = item.get('audio', item.get('audio_path', item.get('path')))
        
        if not emotions_list or not audio_path:
            continue
        
        if isinstance(emotions_list, list):
            from collections import Counter
            emotion_counts = Counter(emotions_list)
            emotion = emotion_counts.most_common(1)[0][0]
        else:
            emotion = emotions_list
        
        emotion_normalized = emotion.capitalize()
        
        if emotion_normalized.startswith('Other'):
            continue
        
        if emotion_normalized in config.emotions:
            full_path = str(data_root / audio_path) if not os.path.isabs(audio_path) else audio_path
            if Path(full_path).exists():
                files_by_emotion[emotion_normalized].append(full_path)
    
    return dict(files_by_emotion)


def load_savee_files(config) -> Dict[str, List[str]]:
    """Load SAVEE dataset files grouped by emotion.

    Filename format: {speaker}_{emotion_code}{number}.wav
    Emotion codes: a=angry, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise
    """
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)

    code_map = {
        'a': 'angry', 'd': 'disgust', 'f': 'fear',
        'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise',
    }

    for wav_file in sorted(data_root.glob('*.wav')):
        name = wav_file.stem
        parts = name.split('_', 1)
        if len(parts) != 2:
            continue
        code_part = parts[1]
        import re
        m = re.match(r'^([a-z]+)', code_part)
        if not m:
            continue
        code = m.group(1)
        emotion = code_map.get(code)
        if emotion and emotion in config.emotions:
            files_by_emotion[emotion].append(str(wav_file))

    return dict(files_by_emotion)


def load_tess_files(config) -> Dict[str, List[str]]:
    """Load TESS dataset files grouped by emotion.

    Directory structure: {data_root}/{speaker}_{emotion}/*.wav
    Speakers: OAF, YAF. Emotions extracted from folder name suffix.
    """
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)

    emo_map = {
        'angry': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
        'pleasant_surprise': 'surprise', 'pleasant_surprised': 'surprise',
    }

    for emo_dir in sorted(data_root.iterdir()):
        if not emo_dir.is_dir():
            continue
        folder_lower = emo_dir.name.lower()
        for suffix, emotion in emo_map.items():
            if folder_lower.endswith(suffix):
                if emotion in config.emotions:
                    wav_files = sorted(emo_dir.glob('*.wav'))
                    files_by_emotion[emotion].extend([str(f) for f in wav_files])
                break

    return dict(files_by_emotion)


def load_meld_files(config) -> Dict[str, List[str]]:
    """Load MELD dataset files grouped by emotion.

    After audio extraction, structure: {data_root}/{emotion}/*.wav
    Same as CAMEO-style loading.
    """
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)

    if not data_root.exists():
        print(f"Warning: MELD audio directory not found: {data_root}")
        return dict(files_by_emotion)

    for emotion_dir in sorted(data_root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name
        if emotion in config.emotions:
            wav_files = sorted(emotion_dir.glob('*.wav'))
            files_by_emotion[emotion].extend([str(f) for f in wav_files])

    return dict(files_by_emotion)


def load_asvp_esd_files(config) -> Dict[str, List[str]]:
    """Load ASVP-ESD dataset files grouped by emotion.

    Filename format: XX-XX-XX-XX-XX-XX-XX-XX-XX[...].wav
    Field index 1 = vocal channel (01=speech, 02=non-speech)
    Field index 2 = emotion code
    Field index 8 = language (01=Chinese, 02=English, 04=French)

    Filters: speech only (01) + English only (02).
    """
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)

    emo_code_map = {
        '02': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry',
        '06': 'fearful', '07': 'disgust', '08': 'surprised',
    }

    for actor_dir in sorted(data_root.iterdir()):
        if not actor_dir.is_dir():
            continue
        for wav_file in sorted(actor_dir.glob('*.wav')):
            parts = wav_file.stem.split('-')
            if len(parts) < 9:
                continue
            vocal_channel = parts[1]
            emo_code = parts[2]
            language = parts[8]
            if vocal_channel != '01' or language != '02':
                continue
            emotion = emo_code_map.get(emo_code)
            if emotion and emotion in config.emotions:
                files_by_emotion[emotion].append(str(wav_file))

    return dict(files_by_emotion)


def load_cameo_files(config) -> Dict[str, List[str]]:
    """Load CAMEO dataset files grouped by emotion.
    
    Directory structure: {data_root}/{emotion}/*.wav
    Only loads emotions listed in config.emotions.
    """
    data_root = Path(config.data_root)
    files_by_emotion = defaultdict(list)
    
    if not data_root.exists():
        print(f"Warning: CAMEO directory not found: {data_root}")
        return dict(files_by_emotion)
    
    for emotion_dir in data_root.iterdir():
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name
        if emotion in config.emotions:
            wav_files = sorted(emotion_dir.glob('*.wav'))
            files_by_emotion[emotion].extend([str(f) for f in wav_files])
    
    return dict(files_by_emotion)


def load_dataset_files(dataset_name: str) -> Dict[str, List[str]]:
    """Load files for the given dataset name."""
    config = DATASET_CONFIGS[dataset_name]
    
    if dataset_name in CAMEO_DATASETS:
        return load_cameo_files(config)
    
    loaders = {
        'esd': load_esd_files,
        'esd_en': lambda cfg: load_esd_files(cfg, speaker_filter=set(range(11, 21))),
        'esd_zh': lambda cfg: load_esd_files(cfg, speaker_filter=set(range(1, 11))),
        'iemocap': load_iemocap_files,
        'ravdess': load_ravdess_files,
        'cremad': load_cremad_files,
        'emodb': load_emodb_files,
        'msp': load_msp_files,
        'savee': load_savee_files,
        'tess': load_tess_files,
        'meld': load_meld_files,
        'asvp_esd': load_asvp_esd_files,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(loaders.keys()) + CAMEO_DATASETS}")
    
    return loaders[dataset_name](config)


def stratified_split(
    files_by_emotion: Dict[str, List[str]],
    train_ratio: float = 0.5,
    val_ratio: float = 0.1,
    seed: int = GLOBAL_SEED
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Stratified split: returns (train, val, test) dicts keyed by emotion."""
    random.seed(seed)
    
    train_files = defaultdict(list)
    val_files = defaultdict(list)
    test_files = defaultdict(list)
    
    for emotion, files in files_by_emotion.items():
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_files[emotion] = shuffled[:n_train]
        val_files[emotion] = shuffled[n_train:n_train + n_val]
        test_files[emotion] = shuffled[n_train + n_val:]
    
    return dict(train_files), dict(val_files), dict(test_files)


def save_split(split_data: Dict[str, List[str]], output_path: Path):
    """Save split data to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    print(f"  Saved to {output_path}")


def prepare_dataset_split(
    dataset_name: str,
    output_dir: Path,
    train_ratio: float = 0.5,
    val_ratio: float = 0.1,
    seed: int = GLOBAL_SEED,
    test_only: bool = False,
):
    """Prepare train/val/test splits for a single dataset.
    
    If test_only is True, all data goes into test.json (for OOD test sets).
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}" + (" [TEST-ONLY]" if test_only else ""))
    print(f"{'='*60}")
    
    files_by_emotion = load_dataset_files(dataset_name)
    
    print(f"\nEmotion distribution:")
    total = 0
    for emotion, files in sorted(files_by_emotion.items()):
        print(f"  {emotion}: {len(files)}")
        total += len(files)
    print(f"  Total: {total}")
    
    if total == 0:
        print(f"Warning: No files found for {dataset_name}, skipping...")
        return
    
    dataset_dir = output_dir / dataset_name
    
    if test_only:
        save_split(files_by_emotion, dataset_dir / 'test.json')
        print(f"\nTest-only mode: all {total} samples -> test.json")
    else:
        train_files, val_files, test_files = stratified_split(
            files_by_emotion, train_ratio, val_ratio, seed
        )
        save_split(train_files, dataset_dir / 'train.json')
        save_split(val_files, dataset_dir / 'val.json')
        save_split(test_files, dataset_dir / 'test.json')
        
        print(f"\nSplit statistics:")
        print(f"  Train: {sum(len(v) for v in train_files.values())} ({train_ratio*100:.0f}%)")
        print(f"  Val:   {sum(len(v) for v in val_files.values())} ({val_ratio*100:.0f}%)")
        print(f"  Test:  {sum(len(v) for v in test_files.values())} ({(1-train_ratio-val_ratio)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='Prepare train/val/test splits for all datasets')
    parser.add_argument('--output-dir', type=str, 
                        default=str(EXP_ROOT / 'data' / 'splits'),
                        help='Output directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.5, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED, help='Random seed')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to process (default: all enabled)')
    parser.add_argument('--cameo', action='store_true',
                        help='Process all CAMEO datasets (test-only mode)')
    parser.add_argument('--test-only', action='store_true',
                        help='Put all data into test.json (no train/val split)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.cameo:
        datasets = CAMEO_DATASETS
        test_only = True  # CAMEO datasets are test-only
    elif args.datasets:
        datasets = args.datasets
        test_only = args.test_only
    else:
        datasets = list(get_enabled_datasets().keys())
        # Exclude CAMEO and Emilia variants; only process core datasets
        datasets = [d for d in datasets if not d.startswith('cameo_') and not d.startswith('emilia')]
        test_only = args.test_only
    
    print("="*60)
    print("Biased Codebook Experiment - Data Split Preparation")
    print("="*60)
    if test_only:
        print(f"Mode:        TEST-ONLY (all data -> test.json)")
    else:
        print(f"Train ratio: {args.train_ratio*100:.0f}%")
        print(f"Val ratio:   {args.val_ratio*100:.0f}%")
        print(f"Test ratio:  {(1-args.train_ratio-args.val_ratio)*100:.0f}%")
    print(f"Output dir:  {output_dir}")
    print(f"Datasets:    {datasets}")
    
    
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Warning: Unknown dataset {dataset_name}, skipping...")
            continue
        if not DATASET_CONFIGS[dataset_name].enabled:
            print(f"Warning: Dataset {dataset_name} is disabled, skipping...")
            continue
        
        prepare_dataset_split(
            dataset_name,
            output_dir,
            args.train_ratio,
            args.val_ratio,
            args.seed,
            test_only=test_only,
        )
    
    print("\n" + "="*60)
    print("All splits prepared successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
