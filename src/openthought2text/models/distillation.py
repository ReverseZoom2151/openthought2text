"""Masked student/teacher distillation utilities for reduced-channel encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReducedChannelDistillationConfig:
    """Weights for matching rich-montage teacher and reduced-montage student."""

    representation_weight: float = 1.0
    logits_weight: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.representation_weight <= 0:
            raise ValueError("representation_weight must be positive")
        if self.logits_weight < 0:
            raise ValueError("logits_weight must be nonnegative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class ReducedChannelDistillationOutput:
    """Component losses and valid-token count for logging/auditing."""

    loss: torch.Tensor
    representation_loss: torch.Tensor
    logits_loss: torch.Tensor | None
    valid_token_count: torch.Tensor


class ReducedChannelDistillationLoss(nn.Module):
    """Match student encoder evidence to a teacher on a shared token mask.

    Teacher tensors are detached by design: this utility trains only the
    reduced-channel student.  Logit matching is optional, but if supplied both
    sides must be aligned ``[batch, tokens, vocabulary]`` tensors.
    """

    def __init__(self, config: ReducedChannelDistillationConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReducedChannelDistillationConfig()

    @staticmethod
    def _validate(
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        token_mask: torch.Tensor,
        student_logits: torch.Tensor | None,
        teacher_logits: torch.Tensor | None,
    ) -> None:
        if student_features.ndim != 3 or teacher_features.ndim != 3:
            raise ValueError("student_features and teacher_features must be [batch, tokens, hidden]")
        if student_features.shape != teacher_features.shape:
            raise ValueError("student_features and teacher_features must have identical shapes")
        if token_mask.shape != student_features.shape[:2]:
            raise ValueError("token_mask must be [batch, tokens]")
        if (student_logits is None) != (teacher_logits is None):
            raise ValueError("student_logits and teacher_logits must be supplied together")
        if student_logits is not None:
            assert teacher_logits is not None
            if student_logits.ndim != 3 or teacher_logits.ndim != 3:
                raise ValueError("student_logits and teacher_logits must be [batch, tokens, vocabulary]")
            if student_logits.shape != teacher_logits.shape:
                raise ValueError("student_logits and teacher_logits must have identical shapes")
            if student_logits.shape[:2] != student_features.shape[:2]:
                raise ValueError("logits must share batch and token axes with features")

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        token_mask: torch.Tensor,
        student_logits: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
    ) -> ReducedChannelDistillationOutput:
        self._validate(student_features, teacher_features, token_mask, student_logits, teacher_logits)
        weights = token_mask.to(dtype=student_features.dtype)
        valid_count = weights.sum()
        denominator = valid_count.clamp_min(1)
        representation_loss = (
            (student_features - teacher_features.detach()).square().mean(dim=-1) * weights
        ).sum() / denominator
        logits_loss: torch.Tensor | None = None
        total = self.config.representation_weight * representation_loss
        if student_logits is not None:
            assert teacher_logits is not None
            temperature = self.config.temperature
            token_kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits.detach() / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1) * temperature**2
            logits_loss = (token_kl * weights).sum() / denominator
            total = total + self.config.logits_weight * logits_loss
        return ReducedChannelDistillationOutput(
            loss=total,
            representation_loss=representation_loss,
            logits_loss=logits_loss,
            valid_token_count=valid_count.to(dtype=torch.long),
        )
