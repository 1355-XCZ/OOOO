# Biased Codebook Experiments for SSL-based Speech Emotion Recognition

This repository investigates how **Residual Vector Quantization (RVQ) codebooks** trained under different conditions affect Speech Emotion Recognition (SER) performance across multiple Self-Supervised Learning (SSL) representations.

The work is organized around four research questions (RQ1--RQ4), each with dedicated evaluation, figure generation, and SLURM submission scripts.

---

## Research Questions

| RQ | Topic | Key Outputs |
|----|-------|-------------|
| **RQ1** | Balanced SSL Codebook Performance | Cosine similarity and SER recall across RVQ layers for emotion2vec, HuBERT, WavLM |
| **RQ2** | Matched SER, Codebook Entropy, and Cross-SSL Comparison | SER F1-Macro, normalized codebook token entropy, LaTeX comparison tables |
| **RQ3** | Effect of Emotion Ratio (3.1) and Annotator Ambiguity (3.2) | Ratio/ambiguity codebook tables, combined layer-wise F1 figures |
| **RQ4** | emotion2vec Codebook Structure Comparison | OOD Macro-F1 table across 9 classification methods and 19+ codebook configurations |

---

## Quick Start

### 1. Environment Setup

```bash
# One command -- works on HPC, local GPU, or CPU-only machines
bash setup_env.sh

# Or with options:
bash setup_env.sh --venv-path /path/to/venv --cuda-version 121
bash setup_env.sh --cpu  # CPU-only mode
```

### 2. Configure Local Paths

After setup, edit `local_config.sh` with your data and model paths:

```bash
export VENV_PATH="/path/to/your/venv"
export DATA_ROOT="/path/to/datasets"          # IEMOCAP, ESD, RAVDESS, CREMA-D, etc.
export E2V_MODEL_PATH="/path/to/emotion2vec_plus_base/model.pt"
```

### 3. Run the Pipeline

Each RQ has a SLURM submission script in `paper_pipeline/slurm/`:

```bash
# Example: submit RQ1 evaluation + figure generation
bash paper_pipeline/slurm/submit_rq1.sh

# Example: submit RQ4 full pipeline (eval -> F1 computation -> table)
bash paper_pipeline/slurm/submit_rq4.sh
```

---

## Project Structure

```
BiasedCodebookExp_v2/
|
|-- configs/                    # Global configuration
|   |-- constants.py            #   Single source of truth (seed, paths, defaults)
|   |-- dataset_config.py       #   Dataset definitions, RVQ/Training configs
|
|-- core/                       # Core library modules
|   |-- config.py               #   Paths, label mappings, seed management
|   |-- features.py             #   SSL feature extraction (emotion2vec, HuBERT, WavLM)
|   |-- quantize.py             #   RVQ codebook loading, cosine similarity, reconstruction
|   |-- standard_rvq_official.py#   RVQ model definitions (StandardRVQ, EncoderDecoderRVQ, NonUniformRVQ)
|   |-- classify.py             #   E2V classification heads and wrappers
|   |-- training.py             #   Dataset, collation, training loop, balanced sampling
|
|-- scripts/
|   |-- train/                  # Codebook and classifier training scripts
|   |   |-- train_balanced_codebook.py
|   |   |-- train_biased_codebook.py
|   |   |-- train_mixed_codebook.py
|   |   |-- train_emilia_codebook.py
|   |   |-- train_ambiguity_codebook.py
|   |   |-- train_ser_classifier.py
|   |-- evaluate/               # Evaluation scripts
|   |   |-- evaluate_unified.py #   Main evaluation entry point (--phase all)
|   |   |-- evaluate_ssl_balanced_2x32.py
|   |-- utils/                  # Data preparation
|       |-- prepare_splits.py
|       |-- prepare_ambiguity_splits.py
|
|-- paper_pipeline/             # Paper figure/table generation pipeline
|   |-- config.py               #   RQ registry and output paths
|   |-- pipeline.py             #   Pipeline orchestration
|   |-- evaluators/             #   Per-RQ evaluation post-processing
|   |   |-- rq2_1_matched_ser.py
|   |   |-- rq2_2_ssl_table.py
|   |   |-- rq2_3_entropy.py
|   |   |-- rq3_1_ratio_table.py
|   |   |-- rq3_1_sample_level.py
|   |   |-- rq3_2_ambiguity_table.py
|   |   |-- rq4_compute_f1.py
|   |-- figures/                #   Per-RQ figure/table rendering
|   |   |-- rq1.py, rq2_*.py, rq3_*.py, rq4.py
|   |-- slurm/                  #   SLURM job scripts and submission helpers
|       |-- env.sh              #   Shared environment setup (auto-detecting paths)
|       |-- submit_rq*.sh       #   One-click submission per RQ
|       |-- *.slurm             #   Individual job definitions
|
|-- setup_env.sh                # One-command environment builder
|-- requirements.txt            # Python dependencies
|-- local_config.sh.template    # Template for local path configuration
|-- .gitignore                  # Excludes codebooks, classifiers, intermediate results
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch + torchaudio | Deep learning framework, audio I/O |
| numpy, scikit-learn | Numerical computation, metrics (F1, accuracy) |
| funasr | emotion2vec feature extraction |
| s3prl | HuBERT / WavLM feature extraction |
| vector-quantize-pytorch | Residual Vector Quantization (lucidrains) |
| matplotlib | Paper figure generation |
| tqdm | Progress bars |

Python 3.10+ is required. See `requirements.txt` for version constraints.

---

## Key Design Decisions

- **Global seed**: All randomness is controlled by `GLOBAL_SEED = 1355` defined in `configs/constants.py`.
- **No hardcoded paths**: All data/model paths are configured via environment variables with fallback defaults. Override in `local_config.sh`.
- **Self-contained**: This repository has no external code dependencies beyond standard pip packages. The RVQ implementation (`core/standard_rvq_official.py`) is included directly.
- **SLURM-native**: All experiments are designed to run as SLURM array jobs with dependency chaining. Submission scripts handle job orchestration automatically.
- **Evaluation methodology**: Emotion-specific codebooks are evaluated on **all** emotions' samples (not oracle-matched), then averaged across codebooks to prevent implicit information leakage.

---

## Datasets

The pipeline supports the following SER datasets (configured in `configs/dataset_config.py`):

**Training / ID evaluation**: ESD (English), IEMOCAP, RAVDESS, CREMA-D

**OOD evaluation**: MSP-Podcast, CAMEO-EMNS, CAMEO-EnterFace, CAMEO-JL-Corpus

Datasets are not included in this repository. Set `DATA_ROOT` in `local_config.sh` to point to your local copies.

---

## Citation

```bibtex
@mastersthesis{zhou2025emotion,
  title={Emotion Information Bottleneck: Precise Rate-Distortion Control for Emotion Recognition},
  author={Zhou, Haoguang},
  school={University of Melbourne},
  year={2025},
  type={Master's Thesis}
}
```

---

## Contact

- **Email**: haoguangz@student.unimelb.edu.au
