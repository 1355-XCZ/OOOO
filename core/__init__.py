"""
BiasedCodebookExp Core Library

Shared utilities extracted from existing evaluation/training scripts.
All code is copied verbatim from the original scripts to ensure reproducibility.

Modules:
    config    - Unified configuration (seed, paths, constants)
    quantize  - RVQ codebook loading, reconstruction, cosine computation
    classify  - E2V classification head, softmax, classify_with_details
    features  - SSL feature extraction (e2v, HuBERT, WavLM)
    datasets  - Dataset loading, splits, emotion mappings
"""
