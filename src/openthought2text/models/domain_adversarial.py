"""Masked cross-subject domain adversary with gradient reversal."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, values: torch.Tensor, scale: float
    ) -> torch.Tensor:
        ctx.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, gradients: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradients, None


def gradient_reverse(values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Identity forward operation that reverses/scales only backward gradients."""
    if scale < 0:
        raise ValueError("gradient reversal scale must be nonnegative")
    return _GradientReversalFunction.apply(values, float(scale))


@dataclass(frozen=True)
class CrossSubjectAdversarialOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    pooled_features: torch.Tensor
    subject_ids: torch.Tensor
    valid_token_counts: torch.Tensor


class CrossSubjectDomainAdversary(nn.Module):
    """Train an encoder to remove subject identity from masked neural evidence.

    ``training_loss`` is the sole API that accepts subject labels.  ``forward``
    accepts only features/masks and is intentionally label-free, so subject
    labels cannot leak into an inference or generation path through this head.
    """

    def __init__(self, hidden_size: int, num_subjects: int) -> None:
        super().__init__()
        if hidden_size < 1 or num_subjects < 2:
            raise ValueError("hidden_size must be positive and num_subjects at least two")
        self.hidden_size = hidden_size
        self.num_subjects = num_subjects
        self.classifier = nn.Linear(hidden_size, num_subjects)

    def _pool(
        self, neural_features: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if neural_features.ndim != 3 or neural_features.shape[-1] != self.hidden_size:
            raise ValueError("neural_features must be [batch, tokens, hidden_size]")
        if token_mask.shape != neural_features.shape[:2]:
            raise ValueError("token_mask must be [batch, tokens]")
        weights = token_mask.to(dtype=neural_features.dtype)
        counts = weights.sum(dim=1)
        if (counts < 1).any():
            raise ValueError("each example needs at least one valid neural token")
        pooled = (neural_features * weights.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        return pooled, counts.to(dtype=torch.long)

    def forward(
        self,
        neural_features: torch.Tensor,
        token_mask: torch.Tensor,
        gradient_reversal_scale: float = 1.0,
    ) -> torch.Tensor:
        """Return subject logits from reversed neural gradients; accepts no IDs."""
        pooled, _ = self._pool(neural_features, token_mask)
        return self.classifier(gradient_reverse(pooled, gradient_reversal_scale))

    def training_loss(
        self,
        neural_features: torch.Tensor,
        token_mask: torch.Tensor,
        subject_ids: torch.Tensor,
        gradient_reversal_scale: float = 1.0,
    ) -> CrossSubjectAdversarialOutput:
        """Explicit train-only domain loss with validated subject labels."""
        pooled, counts = self._pool(neural_features, token_mask)
        if subject_ids.ndim != 1 or subject_ids.shape[0] != neural_features.shape[0]:
            raise ValueError("subject_ids must be [batch]")
        if subject_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("subject_ids must have an integer dtype")
        ids = subject_ids.to(device=neural_features.device, dtype=torch.long)
        if ids.lt(0).any() or ids.ge(self.num_subjects).any():
            raise ValueError("subject_ids must be in [0, num_subjects)")
        logits = self.classifier(gradient_reverse(pooled, gradient_reversal_scale))
        return CrossSubjectAdversarialOutput(
            loss=F.cross_entropy(logits, ids),
            logits=logits,
            pooled_features=pooled,
            subject_ids=ids,
            valid_token_counts=counts,
        )
