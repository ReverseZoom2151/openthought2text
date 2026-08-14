"""Masked neural reconstruction and multi-view consistency objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NeuralSelfSupervisionConfig:
    reconstruction_weight: float = 1.0
    consistency_weight: float = 1.0
    normalize_consistency: bool = True

    def __post_init__(self) -> None:
        if self.reconstruction_weight < 0 or self.consistency_weight < 0:
            raise ValueError("self-supervision loss weights must be nonnegative")
        if self.reconstruction_weight == 0 and self.consistency_weight == 0:
            raise ValueError("at least one self-supervision loss weight must be positive")


class NeuralReconstructionHead(nn.Module):
    """Target-free tokenwise reconstruction prediction from encoder features."""

    def __init__(
        self, hidden_size: int, reconstruction_size: int, bottleneck_size: int | None = None
    ) -> None:
        super().__init__()
        if hidden_size < 1 or reconstruction_size < 1:
            raise ValueError("hidden_size and reconstruction_size must be positive")
        bottleneck_size = bottleneck_size or hidden_size
        if bottleneck_size < 1:
            raise ValueError("bottleneck_size must be positive")
        self.hidden_size = hidden_size
        self.reconstruction_size = reconstruction_size
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, reconstruction_size),
        )

    def forward(self, neural_features: torch.Tensor) -> torch.Tensor:
        if neural_features.ndim != 3 or neural_features.shape[-1] != self.hidden_size:
            raise ValueError("neural_features must be [batch, tokens, hidden_size]")
        return self.network(neural_features)


@dataclass(frozen=True)
class NeuralSelfSupervisionOutput:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    consistency_loss: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_mask: torch.Tensor
    consistency_mask: torch.Tensor
    reconstruction_token_count: torch.Tensor
    consistency_token_count: torch.Tensor


class NeuralReconstructionConsistencyObjective(nn.Module):
    """Neural-only reconstruction plus agreement between two encoder views.

    The first view produces a reconstruction target prediction.  If a second
    view is supplied (for example, from an independently augmented neural
    signal), consistency is applied only where both encoder masks are valid.
    No text/label argument is present in either path.
    """

    def __init__(
        self,
        reconstruction_head: NeuralReconstructionHead,
        config: NeuralSelfSupervisionConfig | None = None,
    ) -> None:
        super().__init__()
        self.reconstruction_head = reconstruction_head
        self.config = config or NeuralSelfSupervisionConfig()

    def forward(
        self,
        primary_features: torch.Tensor,
        primary_mask: torch.Tensor,
        reconstruction_targets: torch.Tensor,
        reconstruction_mask: torch.Tensor | None = None,
        secondary_features: torch.Tensor | None = None,
        secondary_mask: torch.Tensor | None = None,
    ) -> NeuralSelfSupervisionOutput:
        if (
            primary_features.ndim != 3
            or primary_features.shape[-1] != self.reconstruction_head.hidden_size
        ):
            raise ValueError("primary_features must be [batch, tokens, hidden_size]")
        if primary_mask.shape != primary_features.shape[:2]:
            raise ValueError("primary_mask must be [batch, tokens]")
        if reconstruction_targets.shape != (
            *primary_features.shape[:2],
            self.reconstruction_head.reconstruction_size,
        ):
            raise ValueError(
                "reconstruction_targets must align with primary features and reconstruction size"
            )
        valid_primary = primary_mask.bool()
        if reconstruction_mask is None:
            reconstruction_mask = valid_primary
        elif reconstruction_mask.shape != valid_primary.shape:
            raise ValueError("reconstruction_mask must match primary_mask")
        elif (reconstruction_mask.bool() & ~valid_primary).any():
            raise ValueError("reconstruction_mask cannot select padded primary tokens")
        reconstruction_mask = reconstruction_mask.bool()
        if (secondary_features is None) != (secondary_mask is None):
            raise ValueError("secondary_features and secondary_mask must be supplied together")
        if secondary_features is not None:
            assert secondary_mask is not None
            if secondary_features.shape != primary_features.shape:
                raise ValueError("secondary_features must match primary_features")
            if secondary_mask.shape != valid_primary.shape:
                raise ValueError("secondary_mask must match primary_mask")
            consistency_mask = valid_primary & secondary_mask.bool()
        else:
            consistency_mask = torch.zeros_like(valid_primary)
        reconstruction = self.reconstruction_head(primary_features)
        reconstruction_weights = reconstruction_mask.to(dtype=primary_features.dtype)
        reconstruction_count = reconstruction_weights.sum()
        reconstruction_loss = (
            (reconstruction - reconstruction_targets.detach()).square().mean(dim=-1)
            * reconstruction_weights
        ).sum() / reconstruction_count.clamp_min(1)
        if secondary_features is None:
            consistency_loss = primary_features.sum() * 0.0
        else:
            left, right = primary_features, secondary_features
            if self.config.normalize_consistency:
                left, right = F.normalize(left, dim=-1), F.normalize(right, dim=-1)
            consistency_weights = consistency_mask.to(dtype=primary_features.dtype)
            consistency_loss = (
                (left - right).square().mean(dim=-1) * consistency_weights
            ).sum() / consistency_weights.sum().clamp_min(1)
        loss = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.consistency_weight * consistency_loss
        )
        return NeuralSelfSupervisionOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            consistency_loss=consistency_loss,
            reconstruction=reconstruction,
            reconstruction_mask=reconstruction_mask,
            consistency_mask=consistency_mask,
            reconstruction_token_count=reconstruction_count.to(dtype=torch.long),
            consistency_token_count=consistency_mask.sum(dtype=torch.long),
        )
