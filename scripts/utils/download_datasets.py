#!/usr/bin/env python3
"""
Dataset Download & Setup Script

Downloads all required datasets and the emotion2vec model.
Final data is placed into {EXP_ROOT}/data/, matching DATA_ROOT layout.
Temporary archives are staged in {EXP_ROOT}/download/ and cleaned up.

Auto-downloadable:
  - RAVDESS        (Kaggle)
  - CREMA-D        (Kaggle)
  - ESD            (Google Drive)
  - CAMEO subsets   (HuggingFace)
  - emotion2vec    (ModelScope)

License-restricted (manual copy or --copy-from):
  - IEMOCAP        (requires USC license)
  - MSP-Podcast    (requires UTD license)

Usage:
    python scripts/utils/download_datasets.py --all
    python scripts/utils/download_datasets.py --all --copy-from /path/to/existing/data
    python scripts/utils/download_datasets.py --dataset ravdess cremad
    python scripts/utils/download_datasets.py --model
    python scripts/utils/download_datasets.py --verify
"""

import argparse
import glob
import shutil
import subprocess
import sys
import os
import tarfile
import zipfile
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

DATA_DIR = EXP_ROOT / 'data'
STAGING_DIR = EXP_ROOT / 'download'

# Redirect all library caches to STAGING_DIR so we don't fill ~/.cache/
os.environ.setdefault('KAGGLE_CACHE_FOLDER', str(STAGING_DIR / 'kaggle_cache'))
os.environ.setdefault('HF_HOME', str(STAGING_DIR / 'hf_cache'))
os.environ.setdefault('MODELSCOPE_CACHE', str(STAGING_DIR / 'modelscope_cache'))

DATASET_TARGETS = {
    'ravdess':         DATA_DIR / 'RAVDESS',
    'cremad':          DATA_DIR / 'CREMA-D',
    'esd':             DATA_DIR / 'ESD' / 'Emotion Speech Dataset',
    'iemocap':         DATA_DIR / 'IEMOCAP_full_release',
    'msp':             DATA_DIR / 'MSP',
    'cameo_emns':      DATA_DIR / 'CAMEO' / 'emns',
    'cameo_enterface': DATA_DIR / 'CAMEO' / 'enterface',
    'cameo_jl_corpus': DATA_DIR / 'CAMEO' / 'jl_corpus',
}

MODEL_TARGET = DATA_DIR / 'models' / 'emotion2vec_plus_base'

ESD_GDRIVE_ID = '1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v'

# Known archive names in existing data directories
IEMOCAP_ARCHIVES = ['IEMOCAP_full_release.tar.gz', 'IEMOCAP_full_release.tar']
MSP_ARCHIVES = ['msp_podcast.zip']


def _check_exists(path: Path, min_files: int = 5) -> bool:
    if not path.exists():
        return False
    wav_count = len(list(path.rglob('*.wav')))
    return wav_count >= min_files


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Auto-downloadable datasets
# ============================================================

def _kaggle_download(kaggle_handle: str, dataset_key: str, display_name: str):
    """Download a Kaggle dataset directly to the target directory (no ~/.cache)."""
    target = DATASET_TARGETS[dataset_key]
    if _check_exists(target):
        print(f'  [SKIP] {display_name} already exists ({target})')
        return True

    print(f'  Downloading {display_name} from Kaggle ...')
    try:
        import kagglehub
        target.mkdir(parents=True, exist_ok=True)
        dl_path = Path(kagglehub.dataset_download(
            kaggle_handle, output_dir=str(target)))

        if dl_path != target:
            for item in dl_path.rglob('*'):
                if item.is_file():
                    rel = item.relative_to(dl_path)
                    dst = target / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.move(str(item), str(dst))

        wav_count = len(list(target.rglob('*.wav')))
        print(f'  [OK] {display_name}: {wav_count} wav files -> {target}')
        return True
    except Exception as e:
        print(f'  [FAIL] {display_name} download failed: {e}')
        return False


def download_ravdess():
    return _kaggle_download('uwrfkaggler/ravdess-emotional-speech-audio',
                            'ravdess', 'RAVDESS')


def download_cremad():
    return _kaggle_download('ejlok1/cremad', 'cremad', 'CREMA-D')


def download_esd():
    target = DATASET_TARGETS['esd']
    if _check_exists(target):
        print(f'  [SKIP] ESD already exists ({target})')
        return True

    print('  Downloading ESD from Google Drive ...')
    try:
        import gdown
        zip_path = STAGING_DIR / 'ESD.zip'

        if not zip_path.exists():
            url = f'https://drive.google.com/uc?id={ESD_GDRIVE_ID}'
            gdown.download(url, str(zip_path), quiet=False)

        print('  Extracting ESD.zip -> data/ESD/ ...')
        esd_parent = target.parent
        esd_parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(esd_parent)

        zip_path.unlink(missing_ok=True)

        wav_count = len(list(target.rglob('*.wav')))
        print(f'  [OK] ESD: {wav_count} wav files -> {target}')
        print('  Note: Pipeline uses English speakers only (0011-0020)')
        return True
    except Exception as e:
        print(f'  [FAIL] ESD download failed: {e}')
        print(f'  Manual: download from https://drive.google.com/file/d/{ESD_GDRIVE_ID}/view')
        print(f'  Extract to: {target}')
        return False


CAMEO_RAW_DIR = STAGING_DIR / 'cameo_raw'
CAMEO_SPLITS = ['emns', 'enterface', 'jl_corpus']


def _ensure_cameo_raw():
    """Download raw CAMEO dataset once via huggingface-cli (fast with hf_transfer)."""
    marker = CAMEO_RAW_DIR / '.download_complete'
    if marker.exists():
        return True

    print('  Downloading CAMEO raw data via huggingface-cli ...')
    CAMEO_RAW_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    try:
        subprocess.run(
            ['huggingface-cli', 'download', 'amu-cai/CAMEO',
             '--repo-type', 'dataset',
             '--local-dir', str(CAMEO_RAW_DIR)],
            env=env, check=True,
        )
        marker.touch()
        return True
    except FileNotFoundError:
        print('  [WARN] huggingface-cli not found, falling back to datasets lib')
        return False
    except subprocess.CalledProcessError as e:
        print(f'  [FAIL] huggingface-cli download failed (exit {e.returncode})')
        return False


def _extract_cameo_from_parquet(split_name: str):
    """Extract wav files from downloaded CAMEO parquet files."""
    import pyarrow.parquet as pq

    key = f'cameo_{split_name}'
    target = DATASET_TARGETS[key]

    parquet_dir = CAMEO_RAW_DIR / split_name
    pq_files = sorted(parquet_dir.glob('*.parquet')) if parquet_dir.exists() else []
    if not pq_files:
        pq_files = sorted(CAMEO_RAW_DIR.glob(f'{split_name}*.parquet'))
    if not pq_files:
        pq_files = sorted(CAMEO_RAW_DIR.glob(f'**/{split_name}*.parquet'))

    if not pq_files:
        return False

    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for pq_file in pq_files:
        table = pq.read_table(pq_file)
        for i in range(len(table)):
            emotion = table.column('emotion')[i].as_py()
            audio_col = table.column('audio')[i].as_py()

            emo_dir = target / emotion
            emo_dir.mkdir(parents=True, exist_ok=True)
            wav_path = emo_dir / f'{split_name}_{count:05d}.wav'

            if isinstance(audio_col, dict) and 'bytes' in audio_col and audio_col['bytes']:
                wav_path.write_bytes(audio_col['bytes'])
                count += 1
            elif isinstance(audio_col, dict) and 'path' in audio_col:
                src = parquet_dir / audio_col['path'] if (parquet_dir / audio_col['path']).exists() else Path(audio_col['path'])
                if src.exists():
                    shutil.copy2(src, wav_path)
                    count += 1

    return count > 0


def _extract_cameo_fallback(split_name: str):
    """Fallback: use datasets library with decode=False."""
    key = f'cameo_{split_name}'
    target = DATASET_TARGETS[key]

    from datasets import load_dataset, Audio
    ds = load_dataset('amu-cai/CAMEO', split=split_name)
    ds = ds.cast_column('audio', Audio(decode=False))

    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, sample in enumerate(ds):
        emotion = sample['emotion']
        audio = sample['audio']

        emo_dir = target / emotion
        emo_dir.mkdir(parents=True, exist_ok=True)
        wav_path = emo_dir / f'{split_name}_{i:05d}.wav'

        if isinstance(audio, dict) and 'bytes' in audio and audio['bytes']:
            wav_path.write_bytes(audio['bytes'])
            count += 1
        elif isinstance(audio, dict) and 'path' in audio and audio['path']:
            shutil.copy2(audio['path'], wav_path)
            count += 1

    return count > 0


def download_cameo_subset(split_name: str):
    key = f'cameo_{split_name}'
    target = DATASET_TARGETS[key]
    if _check_exists(target):
        print(f'  [SKIP] CAMEO-{split_name} already exists ({target})')
        return True

    print(f'  Processing CAMEO split={split_name} ...')
    try:
        # Strategy 1: extract from pre-downloaded parquet (via huggingface-cli)
        if CAMEO_RAW_DIR.exists() and _extract_cameo_from_parquet(split_name):
            wav_count = len(list(target.rglob('*.wav')))
            print(f'  [OK] CAMEO-{split_name}: {wav_count} wav files -> {target}')
            return True

        # Strategy 2: datasets library with decode=False
        print(f'  Trying datasets library fallback for {split_name} ...')
        if _extract_cameo_fallback(split_name):
            wav_count = len(list(target.rglob('*.wav')))
            print(f'  [OK] CAMEO-{split_name}: {wav_count} wav files -> {target}')
            return True

        print(f'  [FAIL] CAMEO-{split_name}: no audio extracted')
        return False
    except Exception as e:
        print(f'  [FAIL] CAMEO-{split_name} download failed: {e}')
        return False


# ============================================================
# License-restricted datasets (copy from existing location)
# ============================================================

def setup_iemocap(copy_from: str = None):
    target = DATASET_TARGETS['iemocap']
    if _check_exists(target):
        print(f'  [SKIP] IEMOCAP already exists ({target})')
        return True

    if copy_from:
        src = Path(copy_from)

        # Try to find an archive
        for archive_name in IEMOCAP_ARCHIVES:
            archive = src / archive_name
            if archive.exists():
                print(f'  Extracting {archive} -> {target.parent}/ ...')
                target.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, 'r:*') as tf:
                    tf.extractall(target.parent)
                if _check_exists(target):
                    wav_count = len(list(target.rglob('*.wav')))
                    print(f'  [OK] IEMOCAP: {wav_count} wav files -> {target}')
                    return True

        # Try to find the extracted directory
        src_dir = src / 'IEMOCAP_full_release'
        if src_dir.exists() and _check_exists(src_dir):
            print(f'  Copying {src_dir} -> {target} ...')
            shutil.copytree(src_dir, target, dirs_exist_ok=True)
            wav_count = len(list(target.rglob('*.wav')))
            print(f'  [OK] IEMOCAP: {wav_count} wav files -> {target}')
            return True

        print(f'  [FAIL] IEMOCAP not found in {copy_from}')

    print(f'  [MISS] IEMOCAP requires USC license')
    print(f'         Apply: https://sail.usc.edu/iemocap/')
    print(f'         Then:  tar -xzf IEMOCAP_full_release.tar.gz -C {target.parent}')
    return False


def setup_msp(copy_from: str = None):
    target = DATASET_TARGETS['msp']
    if _check_exists(target, min_files=1):
        print(f'  [SKIP] MSP-Podcast already exists ({target})')
        return True

    if copy_from:
        src = Path(copy_from)

        for archive_name in MSP_ARCHIVES:
            archive = src / archive_name
            if archive.exists():
                print(f'  Extracting {archive} -> {target}/ ...')
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive, 'r') as zf:
                    zf.extractall(target)
                if target.exists():
                    json_files = list(target.rglob('*.json'))
                    print(f'  [OK] MSP-Podcast: {len(json_files)} json file(s) -> {target}')
                    return True

        src_dir = src / 'MSP'
        if src_dir.exists():
            print(f'  Copying {src_dir} -> {target} ...')
            shutil.copytree(src_dir, target, dirs_exist_ok=True)
            print(f'  [OK] MSP-Podcast -> {target}')
            return True

        print(f'  [FAIL] MSP not found in {copy_from}')

    print(f'  [MISS] MSP-Podcast requires UTD license')
    print(f'         Apply: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html')
    print(f'         Then place audio + msp_ambigous.json in: {target}')
    return False


# ============================================================
# Model
# ============================================================

def download_model():
    model_pt = MODEL_TARGET / 'model.pt'
    if model_pt.exists():
        print(f'  [SKIP] emotion2vec model already exists ({model_pt})')
        return True

    print('  Downloading emotion2vec_plus_base from ModelScope ...')
    try:
        from modelscope import snapshot_download
        MODEL_TARGET.mkdir(parents=True, exist_ok=True)

        staging_cache = STAGING_DIR / 'modelscope_cache'
        staging_cache.mkdir(parents=True, exist_ok=True)
        model_dir = Path(snapshot_download('iic/emotion2vec_plus_base',
                                           cache_dir=str(staging_cache)))

        # Find and copy model.pt to the final location
        for candidate in [model_dir / 'model.pt'] + list(model_dir.rglob('model.pt')):
            if candidate.exists():
                shutil.copy2(candidate, model_pt)
                break

        # Copy all model files (config, etc.) for completeness
        for f in model_dir.iterdir():
            if f.is_file():
                dst = MODEL_TARGET / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)

        # Clean up staging cache
        shutil.rmtree(staging_cache, ignore_errors=True)

        if model_pt.exists():
            size_mb = model_pt.stat().st_size / (1024 * 1024)
            print(f'  [OK] emotion2vec model ({size_mb:.0f} MB) -> {model_pt}')
            return True
        else:
            print(f'  [WARN] Downloaded but model.pt not found')
            return False
    except Exception as e:
        print(f'  [FAIL] emotion2vec download failed: {e}')
        return False


# ============================================================
# Verify
# ============================================================

def verify_all():
    print('')
    print('=' * 60)
    print(f'  Dataset Verification  (DATA_DIR = {DATA_DIR})')
    print('=' * 60)

    ok_count = 0
    fail_count = 0

    for name, path in DATASET_TARGETS.items():
        min_f = 1 if name == 'msp' else 5
        if _check_exists(path, min_files=min_f):
            wav_count = len(list(path.rglob('*.wav')))
            json_count = len(list(path.rglob('*.json')))
            extra = f'{wav_count} wav' if wav_count else f'{json_count} json'
            print(f'  [OK]   {name:20s}  {extra:>12s}  {path}')
            ok_count += 1
        else:
            print(f'  [MISS] {name:20s}               {path}')
            fail_count += 1

    model_pt = MODEL_TARGET / 'model.pt'
    if model_pt.exists():
        size_mb = model_pt.stat().st_size / (1024 * 1024)
        print(f'  [OK]   {"emotion2vec model":20s}  {size_mb:>8.0f} MB  {model_pt}')
        ok_count += 1
    else:
        print(f'  [MISS] {"emotion2vec model":20s}               {model_pt}')
        fail_count += 1

    print(f'\n  {ok_count} found, {fail_count} missing')
    if fail_count > 0:
        print(f'\n  To download missing: python scripts/utils/download_datasets.py --all')
    return fail_count == 0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download datasets and model for BiasedCodebookExp_v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Target layout (DATA_DIR = {DATA_DIR}):
  data/ESD/Emotion Speech Dataset/    ESD (English speakers 0011-0020)
  data/IEMOCAP_full_release/          IEMOCAP (manual/--copy-from)
  data/RAVDESS/                       RAVDESS
  data/CREMA-D/                       CREMA-D
  data/MSP/                           MSP-Podcast (manual/--copy-from)
  data/CAMEO/emns/                    CAMEO-EMNS
  data/CAMEO/enterface/               CAMEO-EnterFace
  data/CAMEO/jl_corpus/               CAMEO-JL-Corpus
  data/models/emotion2vec_plus_base/  emotion2vec model

Examples:
  %(prog)s --all                               Download all auto-downloadable + model
  %(prog)s --all --copy-from /path/to/data     Also copy IEMOCAP/MSP from existing dir
  %(prog)s --dataset ravdess cremad            Download specific datasets
  %(prog)s --model                             Download emotion2vec model only
  %(prog)s --verify                            Check what is present
''')
    parser.add_argument('--all', action='store_true',
                        help='Download all auto-downloadable datasets + model')
    parser.add_argument('--dataset', nargs='+', default=[],
                        choices=['ravdess', 'cremad', 'esd',
                                 'cameo_emns', 'cameo_enterface', 'cameo_jl_corpus'],
                        help='Download specific dataset(s)')
    parser.add_argument('--model', action='store_true',
                        help='Download emotion2vec_plus_base model')
    parser.add_argument('--copy-from', type=str, default=None,
                        help='Path to existing data directory containing IEMOCAP/MSP archives')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset presence (no downloads)')
    args = parser.parse_args()

    if not any([args.all, args.dataset, args.model, args.verify]):
        parser.print_help()
        return

    if args.verify:
        success = verify_all()
        sys.exit(0 if success else 1)

    _ensure_dirs()

    print('=' * 60)
    print('  BiasedCodebookExp_v2 -- Dataset Download')
    print(f'  Data dir:    {DATA_DIR}')
    print(f'  Staging dir: {STAGING_DIR}')
    if args.copy_from:
        print(f'  Copy from:   {args.copy_from}')
    print('=' * 60)

    results = {}

    auto_downloads = {
        'ravdess': download_ravdess,
        'cremad': download_cremad,
        'esd': download_esd,
        'cameo_emns': lambda: download_cameo_subset('emns'),
        'cameo_enterface': lambda: download_cameo_subset('enterface'),
        'cameo_jl_corpus': lambda: download_cameo_subset('jl_corpus'),
    }

    targets = list(auto_downloads.keys()) if args.all else args.dataset

    # Pre-download CAMEO raw data if any CAMEO subset is requested
    cameo_needed = [t for t in targets if t.startswith('cameo_')]
    if cameo_needed:
        all_cameo_exist = all(_check_exists(DATASET_TARGETS[c]) for c in cameo_needed)
        if not all_cameo_exist:
            print('\n--- cameo (bulk download) ---')
            _ensure_cameo_raw()

    for name in targets:
        print(f'\n--- {name} ---')
        results[name] = auto_downloads[name]()

    if args.all:
        print(f'\n--- iemocap ---')
        results['iemocap'] = setup_iemocap(args.copy_from)
        print(f'\n--- msp ---')
        results['msp'] = setup_msp(args.copy_from)

    if args.all or args.model:
        print('\n--- emotion2vec model ---')
        results['model'] = download_model()

    print('\n' + '=' * 60)
    ok = sum(1 for v in results.values() if v)
    fail = sum(1 for v in results.values() if not v)
    print(f'  Results: {ok} succeeded, {fail} failed/missing')
    print('=' * 60)

    if fail > 0:
        print('\nNext steps for missing datasets:')
        if not results.get('iemocap', True):
            print('  - IEMOCAP: apply at https://sail.usc.edu/iemocap/')
        if not results.get('msp', True):
            print('  - MSP: apply at https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html')
        print(f'\nAfter setup, verify: python scripts/utils/download_datasets.py --verify')
    else:
        print('\nAll datasets ready! Next: python scripts/utils/prepare_splits.py')


if __name__ == '__main__':
    main()
