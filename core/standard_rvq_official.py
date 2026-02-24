"""
Standard RVQ Official Implementation

Uses vector_quantize_pytorch's ResidualVQ as the quantization backend.
Provides StandardRVQOfficial, EncoderDecoderRVQ, and NonUniformRVQ models.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List
from vector_quantize_pytorch import ResidualVQ, VectorQuantize


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


class EncoderDecoderRVQ(nn.Module):
    """
    Encoder-RVQ-Decoder architecture.

    x -> encoder(x) -> RVQ(encoded) -> decoder(quantized) -> reconstructed

    Interface-compatible with StandardRVQOfficial:
      - forward(x, valid_mask=None) -> (reconstructed, indices, commit_loss, stats)
      - quantize_with_layers(x, num_layers) -> quantized
      - get_codebook(layer_idx) -> codebook tensor

    encoder/decoder are nn.Linear(dim, dim, bias=False), initialized to identity.
    RVQ uses EMA codebook updates; encoder/decoder are trained via gradient descent.
    """

    def __init__(self, config: StandardRVQConfig):
        super().__init__()
        self.config = config
        dim = config.feature_dim

        self.encoder = nn.Linear(dim, dim, bias=False)
        self.decoder = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.encoder.weight)
        nn.init.eye_(self.decoder.weight)

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
        )

    def forward(self, x, valid_mask=None):
        """Forward pass (interface-compatible with StandardRVQOfficial)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).float()
            x = x * mask

        encoded = self.encoder(x)
        quantized, indices, commit_loss = self.rvq(encoded)
        reconstructed = self.decoder(quantized)

        if squeeze_output:
            reconstructed = reconstructed.squeeze(1)

        return reconstructed, indices, commit_loss, {}

    def quantize_with_layers(self, x, num_layers: int):
        """Quantize using first `num_layers` layers (with encoder/decoder)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        encoded = self.encoder(x)
        B, T, D = encoded.shape

        residual = encoded.clone()
        quantized = torch.zeros_like(encoded)

        for i in range(min(num_layers, len(self.rvq.layers))):
            layer = self.rvq.layers[i]
            codebook = layer._codebook.embed
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)

            residual_flat = residual.reshape(-1, D)

            if self.config.use_cosine_sim:
                residual_norm = nn.functional.normalize(residual_flat, dim=-1)
                codebook_norm = nn.functional.normalize(codebook, dim=-1)
                sim = torch.matmul(residual_norm, codebook_norm.t())
                idx = sim.argmax(dim=-1)
            else:
                distances = torch.cdist(residual_flat, codebook)
                idx = distances.argmin(dim=-1)

            quantized_layer = codebook[idx].reshape(B, T, D)
            quantized = quantized + quantized_layer
            residual = residual - quantized_layer

        result = self.decoder(quantized)

        if squeeze_output:
            result = result.squeeze(1)

        return result

    def get_codebook(self, layer_idx: int):
        """Get the codebook embedding for the given layer."""
        if layer_idx < len(self.rvq.layers):
            codebook = self.rvq.layers[layer_idx]._codebook.embed
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)
            return codebook
        return None


@dataclass
class NonUniformRVQConfig:
    """Configuration for Non-Uniform RVQ (per-layer codebook sizes)."""
    feature_dim: int = 768
    codebook_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256])
    decay: float = 0.99
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 10
    threshold_ema_dead_code: int = 2
    use_cosine_sim: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.codebook_sizes)

    @property
    def codebook_size(self) -> int:
        """Max codebook size (for compatibility with code that reads this field)."""
        return max(self.codebook_sizes)


class NonUniformRVQ(nn.Module):
    """RVQ with per-layer codebook sizes.

    Each layer is an independent VectorQuantize module, chained via
    residual quantization (each layer quantizes the residual of the previous).
    Interface-compatible with StandardRVQOfficial.
    """

    def __init__(self, config: NonUniformRVQConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            VectorQuantize(
                dim=config.feature_dim,
                codebook_size=k,
                decay=config.decay,
                commitment_weight=config.commitment_weight,
                kmeans_init=config.kmeans_init,
                kmeans_iters=config.kmeans_iters,
                threshold_ema_dead_code=config.threshold_ema_dead_code,
                use_cosine_sim=config.use_cosine_sim,
            )
            for k in config.codebook_sizes
        ])

    def forward(self, x, valid_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).float()
            x = x * mask

        residual = x
        quantized = torch.zeros_like(x)
        all_indices = []
        total_commit_loss = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            q, indices, commit_loss = layer(residual)
            quantized = quantized + q
            residual = residual - q
            all_indices.append(indices)
            total_commit_loss = total_commit_loss + commit_loss

        all_indices = torch.stack(all_indices, dim=-1)

        if squeeze_output:
            quantized = quantized.squeeze(1)

        return quantized, all_indices, total_commit_loss, {}

    def quantize_with_layers(self, x, num_layers: int):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, D = x.shape
        residual = x.clone()
        quantized = torch.zeros_like(x)

        for i in range(min(num_layers, len(self.layers))):
            layer = self.layers[i]
            codebook = layer._codebook.embed
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)

            residual_flat = residual.reshape(-1, D)

            if self.config.use_cosine_sim:
                residual_norm = nn.functional.normalize(residual_flat, dim=-1)
                codebook_norm = nn.functional.normalize(codebook, dim=-1)
                sim = torch.matmul(residual_norm, codebook_norm.t())
                idx = sim.argmax(dim=-1)
            else:
                distances = torch.cdist(residual_flat, codebook)
                idx = distances.argmin(dim=-1)

            quantized_layer = codebook[idx].reshape(B, T, D)
            quantized = quantized + quantized_layer
            residual = residual - quantized_layer

        if squeeze_output:
            quantized = quantized.squeeze(1)
        return quantized

    def get_codebook(self, layer_idx: int):
        if layer_idx < len(self.layers):
            codebook = self.layers[layer_idx]._codebook.embed
            if codebook.dim() == 3:
                codebook = codebook.squeeze(0)
            return codebook
        return None


# Alias for backward compatibility
StandardRVQ = StandardRVQOfficial
