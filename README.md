# Biased Codebook Experiments for SSL-based Speech Emotion Recognition

This repository investigates how **Residual Vector Quantization (RVQ) codebooks** trained under different conditions affect Speech Emotion Recognition (SER) performance across multiple Self-Supervised Learning (SSL) representations.

The pipeline is organized around four research questions (RQ1--RQ4) and produces **3 figures + 1 table** for the paper.

---

## Paper Outputs

| RQ | Output File | Description |
|----|------------|-------------|
| **RQ1** | `rq1_balanced_ssl_ood.png` | Cosine similarity + SER recall across RVQ layers (emotion2vec, HuBERT, WavLM) |
| **RQ2** | `rq2_combined_hubert.png` | SER recall + codebook token entropy (e2v vs HuBERT) |
| **RQ3** | `rq3_combined_js_top2.png` | JS divergence + Top-2 Set Accuracy (ratio × ambiguity) |
| **RQ4** | `rq4_table.tex` / `rq4_ratio_r99_table.tex` | OOD Macro-F1 table (8 codebook sizes, biased 100/0 and mixed 99/1) |

---

## Step 0: Environment Setup

```bash
bash setup_env.sh
pip install -r requirements.txt
```

Then copy and edit the local config (optional -- defaults point to `download/`):

```bash
cp local_config.sh.template local_config.sh
```

### Data Download

Download all auto-downloadable datasets + emotion2vec model:

```bash
python scripts/utils/download_datasets.py --all
```

This downloads to `download/` inside the project. The following are **auto-downloaded**:

| Dataset | Source | Method |
|---------|--------|--------|
| **RAVDESS** | Kaggle | `kagglehub` |
| **CREMA-D** | Kaggle | `kagglehub` |
| **ESD** | Google Drive | `gdown` |
| **CAMEO** (EMNS, EnterFace, JL-Corpus) | HuggingFace | `datasets` |
| **emotion2vec_plus_base** | ModelScope | `modelscope` |

The following require **manual setup** (license-restricted):

| Dataset | How to Obtain |
|---------|--------------|
| **IEMOCAP** | Apply at https://sail.usc.edu/iemocap/ then extract to `download/IEMOCAP_full_release/` |
| **MSP-Podcast** | Apply at https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html then place in `download/MSP/` |

Verify all datasets are present:

```bash
python scripts/utils/download_datasets.py --verify
```

### Quick Environment Check (no data / no GPU)

```bash
python verify_env.py
```

Checks Python version, all pip dependencies, project module imports, RQ registry, and environment variables. Takes a few seconds.

### CPU-Only Smoke Test

```bash
# Run locally:
bash run_smoke_test.sh

# Or submit to a CPU SLURM node:
sbatch paper_pipeline/slurm/smoke_test.slurm
```

Generates synthetic audio, trains a tiny codebook on CPU, validates all module imports, and runs a pipeline dry-run. Takes ~5 minutes, no real data needed.

---

## One-Click Full Reproduction

To reproduce the entire pipeline (training + evaluation + figures) via SLURM dependency chains:

```bash
# Preview all jobs without submitting:
bash paper_pipeline/slurm/reproduce_all.sh --dry-run

# Submit all jobs with correct dependencies:
bash paper_pipeline/slurm/reproduce_all.sh
```

This chains ~600 SLURM tasks in the correct order: data prep → codebook training → evaluation → figure generation. Monitor with `squeue -u $USER`.

---

## Step 1: Data Preparation

```bash
# Create train/val/test splits for all ID datasets (50/10/40, stratified)
python scripts/utils/prepare_splits.py

# Create IEMOCAP ambiguity splits (needed for RQ3)
python scripts/utils/prepare_ambiguity_splits.py

# Create splits for OOD datasets (test-only)
python scripts/utils/prepare_splits.py --cameo --test-only
python scripts/utils/prepare_splits.py --datasets msp --test-only
```

---

## Step 2: Codebook & Classifier Training

All SLURM scripts are in `paper_pipeline/slurm/`. Submit from the project root:

```bash
cd BiasedCodebookExp_v2
```

### RQ1 + RQ2: Balanced & Biased Codebooks (3 SSL models)

Trains balanced + 4 biased codebooks for 4 configs × 4 datasets = 80 tasks:

```bash
sbatch --array=0-79 paper_pipeline/slurm/train_multi_configs.slurm
```

Configs: e2v (2×24), e2v (4×24), HuBERT (1024×24), WavLM (1024×24).

### RQ2: SER Classifiers for HuBERT/WavLM

emotion2vec uses its native classification head. HuBERT and WavLM need trained linear probes:

```bash
# Train classifiers for each SSL × dataset
for SSL in hubert wavlm; do
  for DS in esd_en iemocap ravdess cremad; do
    python scripts/train/train_ser_classifier.py --ssl-model $SSL --dataset $DS
  done
done
```

### RQ3: Mixed-Ratio Codebooks

Trains ratio codebooks (0.95, 0.99) for e2v + HuBERT = 64 tasks:

```bash
sbatch --array=0-63 paper_pipeline/slurm/train_rq3_1_mixed.slurm
```

### RQ4: Multi-Config Codebooks (e2v only)

Trains balanced + biased + mixed_r99 for 8 codebook sizes = 288 tasks:

```bash
sbatch --array=0-287 paper_pipeline/slurm/train_rq4_configs.slurm
```

Sizes: 2×32, 2×64, 2×128, 4×32, 32×8, 64×8, 128×8, 512×16.

---

## Step 3: Evaluation

### RQ1: Balanced Codebook OOD Evaluation

36 tasks (3 SSL × 4 sources × 3 OOD targets):

```bash
sbatch --array=0-35 paper_pipeline/slurm/rq1_eval.slurm
```

### RQ2: SER Recall + Entropy

```bash
# SER Recall: 36 tasks (3 SSL × 12 OOD pairs)
sbatch --array=0-35 paper_pipeline/slurm/rq2_ser_eval.slurm

# Entropy: 36 tasks (3 SSL × 12 OOD pairs)
sbatch --array=0-35 paper_pipeline/slurm/rq2_entropy_eval.slurm
```

### RQ3: Cross-Entropy / Soft-Label Evaluation

6 tasks (2 SSL × 3 sources, test on IEMOCAP):

```bash
sbatch --array=0-5 paper_pipeline/slurm/rq2_ce_eval.slurm
```

Note: Only e2v (0-2) and HuBERT (3-5) are needed for the RQ3 figure.

### RQ4: Unified Method Evaluation

```bash
# Biased (100/0): 32 tasks
sbatch --array=0-31 paper_pipeline/slurm/rq4_eval.slurm

# F1 post-processing (CPU): 32 tasks
sbatch --array=0-31 paper_pipeline/slurm/rq4_compute_f1.slurm

# Mixed ratio (99/1): 32 tasks
sbatch --array=0-31 paper_pipeline/slurm/rq4_ratio_eval.slurm

# Ratio F1 post-processing (CPU): 32 tasks
sbatch --array=0-31 paper_pipeline/slurm/rq4_ratio_f1.slurm
```

---

## Step 4: Figure & Table Generation

All figure/table scripts are CPU-only Python:

```bash
# RQ1 Figure
python -m paper_pipeline.figures.rq1

# RQ2 Figure
python -m paper_pipeline.figures.rq2_combined

# RQ3 Figure
python -m paper_pipeline.figures.rq3_ratio_ambiguity_figure

# RQ4 Tables (biased 100/0 + ratio 99/1)
python -m paper_pipeline.figures.rq4
```

Or use the pipeline orchestrator:

```bash
python -m paper_pipeline.pipeline --rq all --plot
```

Output files are saved to `results/paper_figures_rq/`.

---

## Project Structure

```
BiasedCodebookExp_v2/
├── configs/
│   ├── constants.py              # GLOBAL_SEED=1355, paths, training defaults
│   └── dataset_config.py         # Dataset definitions, RVQ/training configs
│
├── core/                         # Core library
│   ├── config.py                 # Paths, label mappings, seed management
│   ├── features.py               # SSL feature extraction (e2v, HuBERT, WavLM)
│   ├── quantize.py               # RVQ codebook loading, similarity, reconstruction
│   ├── standard_rvq_official.py  # RVQ model definitions
│   ├── classify.py               # Classification heads
│   └── training.py               # Dataset, collation, training loop
│
├── scripts/
│   ├── train/                    # Codebook & classifier training
│   │   ├── train_balanced_codebook.py
│   │   ├── train_biased_codebook.py
│   │   ├── train_mixed_codebook.py
│   │   └── train_ser_classifier.py
│   └── utils/                    # Data preparation & download
│       ├── download_datasets.py  # One-command dataset + model download
│       ├── prepare_splits.py
│       └── prepare_ambiguity_splits.py
│
├── paper_pipeline/               # Paper reproduction pipeline
│   ├── config.py                 # RQ registry (4 entries)
│   ├── pipeline.py               # Orchestrator (--rq, --eval, --plot)
│   ├── evaluators/               # Per-RQ evaluation
│   │   ├── rq1_evaluate.py       # RQ1: per-sample cosine + SER
│   │   ├── rq2_1_matched_ser.py  # RQ2: matched/unmatched SER recall
│   │   ├── rq2_3_entropy.py      # RQ2: codebook token entropy
│   │   ├── rq2_ce.py             # RQ3: CE/JS/Top2 on IEMOCAP soft labels
│   │   ├── rq4_evaluate.py       # RQ4: 9-method evaluation (biased)
│   │   ├── rq4_compute_f1.py     # RQ4: F1 post-processing (biased)
│   │   ├── rq4_ratio_evaluate.py # RQ4: 9-method evaluation (ratio)
│   │   └── rq4_ratio_compute_f1.py
│   ├── figures/                  # Figure/table generation
│   │   ├── rq1.py                # -> rq1_balanced_ssl_ood.png
│   │   ├── rq2_combined.py       # -> rq2_combined_hubert.png
│   │   ├── rq3_ratio_ambiguity_figure.py  # -> rq3_combined_js_top2.png
│   │   └── rq4.py                # -> rq4_table.tex + rq4_ratio_r99_table.tex
│   └── slurm/                    # SLURM job scripts
│       ├── env.sh                # Shared environment (source this first)
│       ├── train_multi_configs.slurm   # RQ1/RQ2 training (80 tasks)
│       ├── train_rq3_1_mixed.slurm    # RQ3 ratio training (64 tasks)
│       ├── train_rq4_configs.slurm    # RQ4 multi-config training (288 tasks)
│       ├── rq1_eval.slurm             # RQ1 evaluation (36 tasks)
│       ├── rq2_ser_eval.slurm         # RQ2 SER evaluation (36 tasks)
│       ├── rq2_entropy_eval.slurm     # RQ2 entropy evaluation (36 tasks)
│       ├── rq2_ce_eval.slurm          # RQ3 CE evaluation (6 tasks)
│       ├── rq4_eval.slurm             # RQ4 biased evaluation (32 tasks)
│       ├── rq4_compute_f1.slurm       # RQ4 F1 post-processing (32 tasks)
│       ├── rq4_ratio_eval.slurm       # RQ4 ratio evaluation (32 tasks)
│       ├── rq4_ratio_f1.slurm         # RQ4 ratio F1 post-processing (32 tasks)
│       ├── smoke_test.slurm           # CPU-only smoke test
│       └── reproduce_all.sh           # One-click full reproduction
│
├── download/                     # Downloaded datasets + model (gitignored)
├── data/splits/                  # Generated by prepare_splits.py (gitignored)
├── codebooks/                    # Trained codebooks (gitignored)
├── classifiers/                  # Trained classifiers (gitignored)
├── results/                      # Evaluation outputs (gitignored)
│   └── paper_figures_rq/         # Final figures/tables (kept in git)
│
├── verify_env.py                 # Quick environment check (no data/GPU)
├── run_smoke_test.sh             # CPU-only smoke test
├── setup_env.sh
├── requirements.txt
├── local_config.sh.template
└── .gitignore
```

---

## Key Design Decisions

- **Global seed**: All randomness controlled by `GLOBAL_SEED = 1355` in `configs/constants.py`.
- **Fair training budget**: Emotion-specific codebooks use all available training samples of their target emotion. Balanced codebooks use total = min(emotion counts), equally sampled per emotion.
- **No hardcoded paths**: All data/model paths configured via environment variables. Override in `local_config.sh`.
- **4 fair emotions**: All evaluations use the 4-class mapping: angry, happy, neutral, sad.
- **OOD evaluation**: RQ1-3 use cross-corpus OOD (leave-one-out among 4 ID datasets). RQ4 uses external OOD (MSP-Podcast + 3 CAMEO datasets).
- **SLURM-native**: All experiments are designed as SLURM array jobs.

---

## Datasets

**Training / ID (4 corpora)**: ESD-EN, IEMOCAP, RAVDESS, CREMA-D

**External OOD (4 test sets, RQ4 only)**: MSP-Podcast, CAMEO-EMNS, CAMEO-EnterFace, CAMEO-JL-Corpus

**Split**: 50% train / 10% val / 40% test, stratified by emotion. CAMEO and MSP are test-only.

Datasets are downloaded to `download/` via `python scripts/utils/download_datasets.py --all`. IEMOCAP and MSP-Podcast require separate license applications.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch + torchaudio | Deep learning, audio I/O |
| numpy, scikit-learn | Metrics (F1, accuracy) |
| funasr | emotion2vec feature extraction |
| s3prl | HuBERT / WavLM feature extraction |
| vector-quantize-pytorch | Residual Vector Quantization |
| matplotlib | Figure generation |
| tqdm | Progress bars |
| gdown, kagglehub | Dataset download (ESD, RAVDESS, CREMA-D) |
| datasets (HuggingFace) | CAMEO download |
| modelscope | emotion2vec model download |

Python 3.10+ required. See `requirements.txt`.

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
