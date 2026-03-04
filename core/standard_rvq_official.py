"""
Standard RVQ Official Implementation

Uses vector_quantize_pytorch's ResidualVQ as the quantization backend.
Codebooks are trained via EMA (no gradient-based codebook updates).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from vector_quantize_pytorch import ResidualVQ


@dataclass
class StandardRVQConfig:
    """Configuration for Standard RVQ"""
    feature_dim: int = 768
    num_layers: int = 8
    codebook_size: int = 128
    decay: float = 0.99
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 10
    threshold_ema_dead_code: int = 2
    use_cosine_sim: bool = False
    learnable_codebook: bool = False
    ema_update: bool = True


class StandardRVQOfficial(nn.Module):
    """Standard RVQ using vector_quantize_pytorch"""

    def __init__(self, config: StandardRVQConfig):
        super().__init__()
        self.config = config

        self.rvq = ResidualVQ(
            dim=config.feature_dim,
            num_quantizers=config.num_layers,
            codebook_size=config.codebook_size,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            kmeans_init=config.kmeans_init,
            kmeans_iters=config.kmeans_iters,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            use_cosine_sim=config.use_cosine_sim,
            learnable_codebook=config.learnable_codebook,
            ema_update=config.ema_update,
        )

    def forward(self, x, valid_mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, D) or (B, T, D)
            valid_mask: Optional bool mask of shape (B, T), True for valid frames.
                        Used to avoid padding frames polluting the codebook.

        Returns:
            quantized: Quantized tensor
            indices: Codebook indices
            commit_loss: Commitment loss
            stats: Empty dict for compatibility
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).float()
            x = x * mask

        quantized, indices, commit_loss = self.rvq(x)

        if squeeze_output:
            quantized = quantized.squeeze(1)

        return quantized, indices, commit_loss, {}

    def quantize_with_layers(self, x, num_layers: int):
        """Quantize using only the first `num_layers` RVQ layers."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = x.shape
        device = x.device

        residual = x.clone()
        quantized = torch.zeros_like(x)

        for i in range(min(num_layers, len(self.rvq.layers))):
            layer = self.rvq.layers[i]
            codebook = layer._codebook.embed
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)

            residual_flat = residual.reshape(-1, D)

            if self.config.use_cosine_sim:
                residual_norm = torch.nn.functional.normalize(residual_flat, dim=-1)
                codebook_norm = torch.nn.functional.normalize(codebook, dim=-1)
                sim = torch.matmul(residual_norm, codebook_norm.t())
                indices = sim.argmax(dim=-1)
            else:
                distances = torch.cdist(residual_flat, codebook)
                indices = distances.argmin(dim=-1)

            quantized_layer = codebook[indices].reshape(B, T, D)
            quantized = quantized + quantized_layer
            residual = residual - quantized_layer

        if squeeze_output:
            quantized = quantized.squeeze(1)

        return quantized

    def get_codebook(self, layer_idx: int):
        """Get the codebook embedding for the given layer."""
        if layer_idx < len(self.rvq.layers):
            codebook = self.rvq.layers[layer_idx]._codebook.embed
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)
            return codebook
        return None



# Alias for backward compatibility
StandardRVQ = StandardRVQOfficial
