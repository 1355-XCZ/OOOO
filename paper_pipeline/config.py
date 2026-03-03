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


_register('1',   'RQ1: Balanced Codebook Cosine Similarity + SER Recall (OOD, 3 SSL configs)',
          evaluator='evaluators.rq1_evaluate',
          figure='figures.rq1')

_register('2',   'RQ2 Combined: SER Recall + Entropy (e2v vs HuBERT/WavLM)',
          figure='figures.rq2_combined')

_register('2.ce', 'RQ2/RQ3: Cross-Entropy Distribution Preservation (IEMOCAP voting, OOD)',
          evaluator='evaluators.rq2_ce',
          figure='figures.rq3_ratio_ambiguity_figure')

_register('4',   'RQ4: e2v Codebook Structure Comparison (OOD Macro-F1)',
          evaluator='evaluators.rq4_compute_f1',
          figure='figures.rq4')
