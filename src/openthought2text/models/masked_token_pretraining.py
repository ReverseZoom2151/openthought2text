"""LaBraM-inspired masked neural-token prediction objective primitives.

This is only a loss and mask-selection utility for RVQ-like neural token IDs;
it does not ship weights or claim a pretrained neural foundation model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import CodebookHealth, codebook_health


@dataclass(frozen=True)
class MaskedNeuralTokenConfig:
    mask_ratio: float = 0.15
    minimum_masked_tokens: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.mask_ratio <= 1:
            raise ValueError("mask_ratio must be in (0, 1]")
        if (
            isinstance(self.minimum_masked_tokens, bool)
            or not isinstance(self.minimum_masked_tokens, int)
            or self.minimum_masked_tokens < 0
        ):
            raise ValueError("minimum_masked_tokens must be a nonnegative integer")


@dataclass(frozen=True)
class MaskedNeuralTokenObjectiveOutput:
    loss: torch.Tensor
    per_level_loss: tuple[torch.Tensor, ...]
    mask_positions: torch.Tensor
    masked_token_count: torch.Tensor
    per_level_health: tuple[CodebookHealth, ...]


def select_mask_positions(
    token_mask: torch.Tensor,
    config: MaskedNeuralTokenConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample valid positions independently per example, never masking padding."""
    if token_mask.ndim != 2:
        raise ValueError("token_mask must be [batch, tokens]")
    valid = token_mask.bool()
    selected = torch.zeros_like(valid)
    for row in range(valid.shape[0]):
        available = valid[row].nonzero(as_tuple=False).squeeze(1)
        count = available.numel()
        if count == 0:
            continue
        requested = max(config.minimum_masked_tokens, int(torch.ceil(torch.tensor(config.mask_ratio * count)).item()))
        requested = min(requested, count)
        if requested == 0:
            continue
        order = torch.randperm(count, device=valid.device, generator=generator)
        selected[row, available[order[:requested]]] = True
    return selected


class MaskedNeuralTokenPredictionObjective(nn.Module):
    """Masked cross-entropy over all RVQ codebook levels at selected tokens."""

    def __init__(self, config: MaskedNeuralTokenConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskedNeuralTokenConfig()

    @staticmethod
    def _validate(
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        mask_positions: torch.Tensor | None,
    ) -> None:
        if token_ids.ndim != 3:
            raise ValueError("token_ids must be [batch, tokens, levels]")
        if token_logits.ndim != 4:
            raise ValueError("token_logits must be [batch, tokens, levels, codebook_size]")
        if token_logits.shape[:3] != token_ids.shape:
            raise ValueError("token_logits and token_ids must share batch, token, and level axes")
        if token_mask.shape != token_ids.shape[:2]:
            raise ValueError("token_mask must be [batch, tokens]")
        if mask_positions is not None:
            if mask_positions.shape != token_mask.shape:
                raise ValueError("mask_positions must match token_mask")
            if (mask_positions.bool() & ~token_mask.bool()).any():
                raise ValueError("mask_positions cannot select padded tokens")
        codebook_size = token_logits.shape[-1]
        if codebook_size < 2:
            raise ValueError("token_logits needs at least two codebook classes")
        if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("token_ids must have an integer dtype")
        if token_ids.lt(0).any() or token_ids.ge(codebook_size).any():
            raise ValueError("token_ids contain a code outside token_logits vocabulary")

    def forward(
        self,
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskedNeuralTokenObjectiveOutput:
        self._validate(token_logits, token_ids, token_mask, mask_positions)
        selected = (
            select_mask_positions(token_mask, self.config, generator)
            if mask_positions is None
            else mask_positions.bool()
        )
        weights = selected.to(dtype=token_logits.dtype)
        denominator = weights.sum().clamp_min(1)
        per_level: list[torch.Tensor] = []
        health: list[CodebookHealth] = []
        for level in range(token_ids.shape[-1]):
            logits = token_logits[:, :, level, :]
            ids = token_ids[:, :, level].to(dtype=torch.long)
            token_loss = F.cross_entropy(logits.flatten(0, 1), ids.flatten(), reduction="none").view_as(weights)
            per_level.append((token_loss * weights).sum() / denominator)
            health.append(codebook_health(ids, token_logits.shape[-1], selected))
        if per_level:
            loss = torch.stack(per_level).mean()
        else:  # unreachable after shape validation, retained as a safe invariant.
            loss = token_logits.sum() * 0.0
        return MaskedNeuralTokenObjectiveOutput(
            loss=loss,
            per_level_loss=tuple(per_level),
            mask_positions=selected,
            masked_token_count=selected.sum(dtype=torch.long),
            per_level_health=tuple(health),
        )
