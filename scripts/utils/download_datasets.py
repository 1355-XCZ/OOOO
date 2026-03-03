#!/usr/bin/env python3
"""
Dataset Download & Setup Script

Downloads all required datasets and the emotion2vec model into
{EXP_ROOT}/download/, matching the directory layout expected by DATA_ROOT.

Auto-downloadable:
  - RAVDESS        (Kaggle)
  - CREMA-D        (Kaggle)
  - ESD            (Google Drive)
  - CAMEO subsets   (HuggingFace)
  - emotion2vec    (ModelScope)

Manual (instructions printed):
  - IEMOCAP        (requires USC license)
  - MSP-Podcast    (requires UTD license)

Usage:
    python scripts/utils/download_datasets.py --all
    python scripts/utils/download_datasets.py --dataset ravdess cremad
    python scripts/utils/download_datasets.py --model
    python scripts/utils/download_datasets.py --verify
"""

import argparse
import shutil
import sys
import os
import zipfile
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

DOWNLOAD_DIR = EXP_ROOT / 'download'

DATASET_TARGETS = {
    'ravdess':  DOWNLOAD_DIR / 'RAVDESS',
    'cremad':   DOWNLOAD_DIR / 'CREMA-D',
    'esd':      DOWNLOAD_DIR / 'ESD' / 'Emotion Speech Dataset',
    'iemocap':  DOWNLOAD_DIR / 'IEMOCAP_full_release',
    'msp':      DOWNLOAD_DIR / 'MSP',
    'cameo_emns':      DOWNLOAD_DIR / 'CAMEO' / 'emns',
    'cameo_enterface': DOWNLOAD_DIR / 'CAMEO' / 'enterface',
    'cameo_jl_corpus': DOWNLOAD_DIR / 'CAMEO' / 'jl_corpus',
}

MODEL_TARGET = DOWNLOAD_DIR / 'models' / 'emotion2vec_plus_base'

ESD_GDRIVE_ID = '1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v'


def _check_exists(path: Path, min_files: int = 5) -> bool:
    if not path.exists():
        return False
    wav_count = len(list(path.rglob('*.wav')))
    return wav_count >= min_files


def download_ravdess():
    target = DATASET_TARGETS['ravdess']
    if _check_exists(target):
        print(f'  [SKIP] RAVDESS already exists ({target})')
        return True

    print('  Downloading RAVDESS from Kaggle ...')
    try:
        import kagglehub
        cache_path = kagglehub.dataset_download('uwrfkaggler/ravdess-emotional-speech-audio')
        cache_path = Path(cache_path)
        target.mkdir(parents=True, exist_ok=True)

        for item in cache_path.rglob('*'):
            if item.is_file():
                rel = item.relative_to(cache_path)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        wav_count = len(list(target.rglob('*.wav')))
        print(f'  [OK] RAVDESS: {wav_count} wav files -> {target}')
        return True
    except Exception as e:
        print(f'  [FAIL] RAVDESS download failed: {e}')
        return False


def download_cremad():
    target = DATASET_TARGETS['cremad']
    if _check_exists(target):
        print(f'  [SKIP] CREMA-D already exists ({target})')
        return True

    print('  Downloading CREMA-D from Kaggle ...')
    try:
        import kagglehub
        cache_path = kagglehub.dataset_download('ejlok1/cremad')
        cache_path = Path(cache_path)
        target.mkdir(parents=True, exist_ok=True)

        for item in cache_path.rglob('*'):
            if item.is_file():
                rel = item.relative_to(cache_path)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        wav_count = len(list(target.rglob('*.wav')))
        print(f'  [OK] CREMA-D: {wav_count} wav files -> {target}')
        return True
    except Exception as e:
        print(f'  [FAIL] CREMA-D download failed: {e}')
        return False


def download_esd():
    target = DATASET_TARGETS['esd']
    if _check_exists(target):
        print(f'  [SKIP] ESD already exists ({target})')
        return True

    print('  Downloading ESD from Google Drive ...')
    try:
        import gdown
        target.parent.mkdir(parents=True, exist_ok=True)
        zip_path = target.parent / 'ESD.zip'

        if not zip_path.exists():
            url = f'https://drive.google.com/uc?id={ESD_GDRIVE_ID}'
            gdown.download(url, str(zip_path), quiet=False)

        print('  Extracting ESD.zip ...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target.parent)

        if zip_path.exists():
            zip_path.unlink()

        wav_count = len(list(target.rglob('*.wav')))
        print(f'  [OK] ESD: {wav_count} wav files -> {target}')
        print('  Note: Pipeline uses English speakers only (0011-0020)')
        return True
    except Exception as e:
        print(f'  [FAIL] ESD download failed: {e}')
        print(f'  Manual: download from https://drive.google.com/file/d/{ESD_GDRIVE_ID}/view')
        print(f'  Extract to: {target}')
        return False


def download_cameo_subset(split_name: str):
    key = f'cameo_{split_name}'
    target = DATASET_TARGETS[key]
    if _check_exists(target):
        print(f'  [SKIP] CAMEO-{split_name} already exists ({target})')
        return True

    print(f'  Downloading CAMEO split={split_name} from HuggingFace ...')
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np

        ds = load_dataset('amu-cai/CAMEO', split=split_name)
        target.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, sample in enumerate(ds):
            emotion = sample['emotion']
            audio = sample['audio']
            sr = audio['sampling_rate']
            array = np.array(audio['array'], dtype=np.float32)

            emo_dir = target / emotion
            emo_dir.mkdir(parents=True, exist_ok=True)
            wav_path = emo_dir / f'{split_name}_{i:05d}.wav'
            sf.write(str(wav_path), array, sr)
            count += 1

        print(f'  [OK] CAMEO-{split_name}: {count} wav files -> {target}')
        return True
    except Exception as e:
        print(f'  [FAIL] CAMEO-{split_name} download failed: {e}')
        return False


def download_model():
    model_pt = MODEL_TARGET / 'model.pt'
    if model_pt.exists():
        print(f'  [SKIP] emotion2vec model already exists ({model_pt})')
        return True

    print('  Downloading emotion2vec_plus_base from ModelScope ...')
    try:
        from modelscope import snapshot_download
        MODEL_TARGET.mkdir(parents=True, exist_ok=True)
        model_dir = snapshot_download('iic/emotion2vec_plus_base',
                                      cache_dir=str(MODEL_TARGET.parent))
        model_dir = Path(model_dir)

        if not model_pt.exists():
            candidate = model_dir / 'model.pt'
            if candidate.exists():
                shutil.copy2(candidate, model_pt)
            else:
                for f in model_dir.rglob('model.pt'):
                    shutil.copy2(f, model_pt)
                    break

        if model_pt.exists():
            print(f'  [OK] emotion2vec model -> {model_pt}')
            return True
        else:
            print(f'  [WARN] Downloaded but model.pt not found at expected location')
            print(f'         Check: {model_dir}')
            return False
    except Exception as e:
        print(f'  [FAIL] emotion2vec download failed: {e}')
        return False


def print_manual_instructions():
    print('')
    print('=' * 60)
    print('  Manual Dataset Setup (license-restricted)')
    print('=' * 60)

    iemocap_target = DATASET_TARGETS['iemocap']
    msp_target = DATASET_TARGETS['msp']

    print(f'''
  IEMOCAP:
    1. Apply at https://sail.usc.edu/iemocap/
    2. After approval, download IEMOCAP_full_release.tar.gz
    3. Extract to: {iemocap_target}
       tar -xzf IEMOCAP_full_release.tar.gz -C {iemocap_target.parent}

  MSP-Podcast:
    1. Apply at https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html
    2. After approval, download and prepare the dataset
    3. Place audio + msp_ambigous.json in: {msp_target}
       Expected structure:
         {msp_target}/
           msp_ambigous.json     # metadata with emotion annotations
           <audio_files>         # wav files referenced in the JSON
''')


def verify_all():
    print('')
    print('=' * 60)
    print('  Dataset Verification')
    print('=' * 60)

    ok_count = 0
    fail_count = 0

    for name, path in DATASET_TARGETS.items():
        if _check_exists(path):
            wav_count = len(list(path.rglob('*.wav')))
            print(f'  [OK]   {name:20s}  {wav_count:>6d} wav files  {path}')
            ok_count += 1
        else:
            print(f'  [MISS] {name:20s}  NOT FOUND  {path}')
            fail_count += 1

    model_pt = MODEL_TARGET / 'model.pt'
    if model_pt.exists():
        size_mb = model_pt.stat().st_size / (1024 * 1024)
        print(f'  [OK]   {"emotion2vec model":20s}  {size_mb:.0f} MB     {model_pt}')
        ok_count += 1
    else:
        print(f'  [MISS] {"emotion2vec model":20s}  NOT FOUND  {model_pt}')
        fail_count += 1

    print(f'\n  {ok_count} found, {fail_count} missing')
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets and model for BiasedCodebookExp_v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --all              Download everything auto-downloadable + model
  %(prog)s --dataset ravdess  Download only RAVDESS
  %(prog)s --model            Download only emotion2vec model
  %(prog)s --verify           Check which datasets are present
''')
    parser.add_argument('--all', action='store_true',
                        help='Download all auto-downloadable datasets + model')
    parser.add_argument('--dataset', nargs='+', default=[],
                        choices=['ravdess', 'cremad', 'esd',
                                 'cameo_emns', 'cameo_enterface', 'cameo_jl_corpus'],
                        help='Download specific dataset(s)')
    parser.add_argument('--model', action='store_true',
                        help='Download emotion2vec_plus_base model')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset presence (no downloads)')
    args = parser.parse_args()

    if not any([args.all, args.dataset, args.model, args.verify]):
        parser.print_help()
        return

    if args.verify:
        success = verify_all()
        sys.exit(0 if success else 1)

    print('=' * 60)
    print('  BiasedCodebookExp_v2 -- Dataset Download')
    print(f'  Target: {DOWNLOAD_DIR}')
    print('=' * 60)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    download_map = {
        'ravdess': download_ravdess,
        'cremad': download_cremad,
        'esd': download_esd,
        'cameo_emns': lambda: download_cameo_subset('emns'),
        'cameo_enterface': lambda: download_cameo_subset('enterface'),
        'cameo_jl_corpus': lambda: download_cameo_subset('jl_corpus'),
    }

    targets = list(download_map.keys()) if args.all else args.dataset
    for name in targets:
        print(f'\n--- {name} ---')
        results[name] = download_map[name]()

    if args.all or args.model:
        print('\n--- emotion2vec model ---')
        results['model'] = download_model()

    if args.all:
        print_manual_instructions()

    print('\n' + '=' * 60)
    ok = sum(1 for v in results.values() if v)
    fail = sum(1 for v in results.values() if not v)
    print(f'  Download Results: {ok} succeeded, {fail} failed')
    print('=' * 60)

    if args.all:
        print('\nNext steps:')
        print('  1. Set up IEMOCAP and MSP-Podcast manually (see above)')
        print('  2. Verify:  python scripts/utils/download_datasets.py --verify')
        print('  3. Run:     python scripts/utils/prepare_splits.py')


if __name__ == '__main__':
    main()
