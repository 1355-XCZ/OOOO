"""
Paper Pipeline -- RQ-based Configuration

Maps each Research Question to its evaluation and figure generation modules.
Reuses paths and constants from core.config and scripts/reproduce/config.
"""

from pathlib import Path
from collections import OrderedDict

# ============================================================
# Paths
# ============================================================
PIPELINE_DIR = Path(__file__).resolve().parent          # paper_pipeline/
EXP_ROOT = PIPELINE_DIR.parent                          # BiasedCodebookExp_v2/

CODEBOOK_DIR = EXP_ROOT / 'codebooks'
RESULTS_DIR = EXP_ROOT / 'results'
SPLITS_DIR = EXP_ROOT / 'data' / 'splits'
PAPER_FIGURES_DIR = RESULTS_DIR / 'paper_figures_rq'

# ============================================================
# Core experiment parameters
# ============================================================
ID_DATASETS = ['esd_en', 'iemocap', 'ravdess', 'cremad']

DATASET_DISPLAY = {
    'esd_en': 'ESD-EN', 'iemocap': 'IEMOCAP',
    'ravdess': 'RAVDESS', 'cremad': 'CREMA-D',
}

CB_CONFIG = '128x8'
NUM_LAYERS = 8

# ============================================================
# RQ Registry
#
# Each entry:  rq_id -> {evaluator_module, figure_module, description}
# Modules are lazily imported by pipeline.py.
# ============================================================

RQ_REGISTRY = OrderedDict()


def _register(rq_id: str, description: str,
              evaluator: str = None, figure: str = None):
    RQ_REGISTRY[rq_id] = {
        'description': description,
        'evaluator': evaluator,
        'figure': figure,
    }


_register('1',   'RQ1: Balanced Codebook L2 + SER Recall (OOD, 4 configs)',
          figure='figures.rq1')

_register('2.1', 'RQ2.1: SER-F1 -- Matched vs Unmatched vs Balanced (e2v, OOD)',
          evaluator='evaluators.rq2_1_matched_ser',
          figure='figures.rq2_1')

_register('2.2', 'RQ2.2: SSL comparison table (L2 + SER-F1, 128x8, Layer 8, OOD)',
          evaluator='evaluators.rq2_2_ssl_table',
          figure='figures.rq2_2')

_register('2.3', 'RQ2.3: Codebook Token Entropy (OOD, 128x8, e2v)',
          evaluator='evaluators.rq2_3_entropy',
          figure='figures.rq2_3')

_register('2', 'RQ2 Combined: SER-F1 + Entropy (e2v vs HuBERT/WavLM)',
          figure='figures.rq2_combined')

_register('3.1', 'RQ3.1: Emotion Ratio Codebook (Cosine Sim + SER F1-Macro)',
          evaluator='evaluators.rq3_1_ratio_table',
          figure='figures.rq3_1')

_register('3.2', 'RQ3.2: Annotator Ambiguity Codebook (Cosine Sim + SER F1-Macro)',
          evaluator='evaluators.rq3_2_ambiguity_table',
          figure='figures.rq3_2')

_register('4',   'RQ4: e2v Codebook Structure Comparison (OOD Macro-F1)',
          evaluator='evaluators.rq4_compute_f1',
          figure='figures.rq4')
