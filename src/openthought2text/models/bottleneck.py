"""Auditable optional discrete bottleneck for neural token representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .residual_quantizer import ResidualVectorQuantizer
from .tokenizer import CodebookHealth


@dataclass(frozen=True)
class NeuralRepresentationBottleneckOutput:
    """Continuous and discrete representations with quantizer audit metadata."""

    continuous_features: torch.Tensor
    quantized_features: torch.Tensor
    mask: torch.Tensor
    indices: torch.Tensor | None
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    per_level_health: tuple[CodebookHealth, ...]

    @property
    def loss(self) -> torch.Tensor:
        return self.commitment_loss + self.codebook_loss

    @property
    def features(self) -> torch.Tensor:
        """The representation selected for discrete-bottleneck ablations."""
        return self.quantized_features

    @property
    def is_quantized(self) -> bool:
        return self.indices is not None


class NeuralRepresentationBottleneck(nn.Module):
    """Expose optional residual quantization without obscuring continuous input.

    Padded token features are zeroed before either branch.  Consequently they
    cannot influence valid vectors, VQ assignments, losses, or health metrics.
    When no quantizer is configured this is a transparent masked identity with
    a differentiable zero auxiliary loss.
    """

    def __init__(self, hidden_size: int, quantizer: ResidualVectorQuantizer | None = None) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if quantizer is not None and quantizer.config.embedding_dim != hidden_size:
            raise ValueError("quantizer embedding_dim must match hidden_size")
        self.hidden_size = hidden_size
        self.quantizer = quantizer

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> NeuralRepresentationBottleneckOutput:
        if features.ndim != 3 or features.shape[-1] != self.hidden_size:
            raise ValueError("features must be [batch, tokens, hidden_size]")
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must be [batch, tokens]")
        valid = mask.bool()
        continuous = features * valid.unsqueeze(-1).to(features.dtype)
        if self.quantizer is None:
            zero = continuous.sum() * 0.0
            return NeuralRepresentationBottleneckOutput(
                continuous_features=continuous,
                quantized_features=continuous,
                mask=valid,
                indices=None,
                commitment_loss=zero,
                codebook_loss=zero,
                per_level_health=(),
            )
        quantized = self.quantizer(continuous, valid)
        return NeuralRepresentationBottleneckOutput(
            continuous_features=continuous,
            quantized_features=quantized.quantized,
            mask=valid,
            indices=quantized.indices,
            commitment_loss=quantized.commitment_loss,
            codebook_loss=quantized.codebook_loss,
            per_level_health=quantized.per_level_health,
        )
