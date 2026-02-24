"""
RVQ Codebook Loading, Reconstruction, and Cosine Computation

Provides:
    - load_codebook: load a trained RVQ checkpoint
    - compute_cosine / compute_similarity: frame-averaged similarity metrics
    - get_all_reconstructions: efficient layerwise RVQ reconstruction
    - Re-exported RVQ model classes from standard_rvq_official
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .standard_rvq_official import (
    StandardRVQOfficial, StandardRVQConfig,
    EncoderDecoderRVQ, NonUniformRVQ, NonUniformRVQConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# Codebook loading  (from evaluate_comprehensive_2x32.py L116-149)
# ============================================================

def load_codebook(codebook_path: str, device: str = 'cuda') -> Optional[nn.Module]:
    """Load RVQ codebook, auto-detect model type (standard / encoder_decoder)."""
    p = Path(codebook_path)
    if not p.exists():
        logger.warning(f"Codebook not found: {codebook_path}")
        return None

    checkpoint = torch.load(codebook_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})
    model_type = checkpoint.get('model_type', 'standard')

    if model_type == 'non_uniform':
        if isinstance(cfg, dict):
            config = NonUniformRVQConfig(
                feature_dim=cfg.get('feature_dim', 768),
                codebook_sizes=cfg.get('codebook_sizes', [2, 4, 8, 16, 32, 64, 128, 256]),
                use_cosine_sim=cfg.get('use_cosine_sim', False),
            )
        else:
            config = cfg
        model = NonUniformRVQ(config)
    else:
        if isinstance(cfg, dict):
            config = StandardRVQConfig(
                feature_dim=cfg.get('feature_dim', 768),
                num_layers=cfg.get('num_layers', 32),
                codebook_size=cfg.get('codebook_size', 2),
                use_cosine_sim=cfg.get('use_cosine_sim', False),
            )
        else:
            config = cfg

        if model_type == 'encoder_decoder':
            model = EncoderDecoderRVQ(config)
        else:
            model = StandardRVQOfficial(config)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).eval()
    logger.info(f"  Loaded {p.name} (type={model_type}, L={config.num_layers}, K={config.codebook_size})")
    return model


# ============================================================
# Cosine similarity  (from evaluate_comprehensive_2x32.py L183-189)
# ============================================================

def compute_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frame-averaged cosine similarity."""
    if a.dim() == 2:
        a = a.mean(dim=0)
    if b.dim() == 2:
        b = b.mean(dim=0)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_similarity(a: torch.Tensor, b: torch.Tensor, metric: str = 'cosine') -> float:
    """Frame-averaged similarity / distance score.

    For cosine: returns cosine similarity in [-1, 1] (higher = more similar).
    For l2: returns L2 distance >= 0 (lower = more similar).
    """
    if a.dim() == 2:
        a = a.mean(dim=0)
    if b.dim() == 2:
        b = b.mean(dim=0)
    if metric == 'cosine':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    elif metric == 'l2':
        return torch.dist(a, b, p=2).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================
# Layerwise reconstruction  (from evaluate_comprehensive_2x32.py L192-231)
# ============================================================

def get_all_reconstructions(
    model: nn.Module,
    features: torch.Tensor,
    layers: List[int],
    device: str = 'cuda',
) -> Dict[int, torch.Tensor]:
    """
    Efficiently compute reconstructions at all specified layers.
    One forward pass, incremental accumulation.

    Args:
        model:    RVQ codebook model (StandardRVQOfficial or EncoderDecoderRVQ)
        features: Input features tensor [T, D] or [1, T, D]
        layers:   List of layer numbers (1-indexed, cumulative)
        device:   'cuda' or 'cpu'

    Returns:
        {layer_number: reconstructed_features_tensor}
    """
    if features.dim() == 2:
        features = features.unsqueeze(0)
    features = features.to(device)

    is_enc_dec = isinstance(model, EncoderDecoderRVQ)
    is_nu = isinstance(model, NonUniformRVQ)
    num_layers = model.config.num_layers
    max_layer = min(max(layers), num_layers)

    with torch.no_grad():
        if is_enc_dec:
            encoded = model.encoder(features)
            _, indices, _ = model.rvq(encoded)
            partial = torch.zeros_like(encoded)
        else:
            _, indices, _, _ = model(features)
            partial = torch.zeros_like(features)

        def _get_codebook_embed(layer_idx):
            if is_nu:
                return model.layers[layer_idx]._codebook.embed[0]
            else:
                return model.rvq.layers[layer_idx]._codebook.embed[0]

        results = {}
        for l in range(max_layer):
            layer_indices = indices[:, :, l]
            codebook = _get_codebook_embed(l)
            partial = partial + codebook[layer_indices]

            if (l + 1) in layers:
                if is_enc_dec:
                    results[l + 1] = model.decoder(partial.clone()).squeeze(0)
                else:
                    results[l + 1] = partial.clone().squeeze(0)

    return results
