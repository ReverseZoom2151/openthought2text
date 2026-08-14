"""Small supervised training loop for the composed neural-to-text model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from openthought2text.data import NeuralTensorBatch
from openthought2text.models import NeuralToTextModel, NeuralToTextTrainingOutput


@dataclass(frozen=True)
class SupervisedStepResult:
    """Numerical record of one auditable optimization step."""

    loss: float
    gradient_norm: float | None
    batch_size: int
    output: NeuralToTextTrainingOutput


def supervised_train_step(
    model: NeuralToTextModel,
    batch: NeuralTensorBatch,
    target_ids: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    labels: torch.Tensor | None = None,
    coordinates: torch.Tensor | None = None,
    anchor_targets: torch.Tensor | None = None,
    anchor_position_mask: torch.Tensor | None = None,
    sample_rate_hz: float = 200.0,
    gradient_clip_norm: float | None = 1.0,
) -> SupervisedStepResult:
    """Run one teacher-forced update; generation is never called here."""

    if target_ids.ndim != 2 or target_ids.shape[0] != batch.signals.shape[0]:
        raise ValueError("target_ids must have one sequence per neural batch row")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch.signals,
        target_ids,
        labels=labels,
        channel_mask=batch.channel_mask,
        coordinates=coordinates,
        token_mask=batch.time_mask,
        anchor_targets=anchor_targets,
        anchor_position_mask=anchor_position_mask,
        sample_rate_hz=sample_rate_hz,
    )
    loss = output.loss
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite supervised loss")
    loss.backward()
    gradient_norm = None
    if gradient_clip_norm is not None:
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm))
    optimizer.step()
    return SupervisedStepResult(
        loss=float(loss.detach()),
        gradient_norm=gradient_norm,
        batch_size=batch.signals.shape[0],
        output=output,
    )
