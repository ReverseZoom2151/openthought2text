"""Evidence-factorized candidate scoring with validation-fitted frozen weights."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ValidationFittedScoreWeights:
    """Immutable linear score weights fitted only from a validation table."""

    lambda_neural: float
    lambda_lm: float
    lambda_length: float
    validation_examples: int
    ridge_regularization: float
    fit_method: str = "ridge_least_squares"

    def __post_init__(self) -> None:
        for name, value in {
            "lambda_neural": self.lambda_neural,
            "lambda_lm": self.lambda_lm,
            "lambda_length": self.lambda_length,
            "ridge_regularization": self.ridge_regularization,
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.validation_examples < 3:
            raise ValueError("validation_examples must be at least three")
        if self.ridge_regularization <= 0:
            raise ValueError("ridge_regularization must be positive")
        if self.fit_method != "ridge_least_squares":
            raise ValueError("fit_method must record ridge_least_squares")


def fit_factorized_score_weights(
    validation_neural_scores: torch.Tensor,
    validation_lm_scores: torch.Tensor,
    validation_length_scores: torch.Tensor,
    validation_utilities: torch.Tensor,
    ridge_regularization: float = 1e-4,
) -> ValidationFittedScoreWeights:
    """Fit immutable weights from explicit scalar validation utilities.

    This routine is intentionally separate from inference.  ``validation_utilities``
    are caller-defined held-out utilities (for example, candidate correctness),
    never accepted by the scorer's inference API.
    """
    inputs = {
        "validation_neural_scores": validation_neural_scores,
        "validation_lm_scores": validation_lm_scores,
        "validation_length_scores": validation_length_scores,
        "validation_utilities": validation_utilities,
    }
    for name, value in inputs.items():
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional validation tensor")
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    count = validation_utilities.numel()
    if count < 3:
        raise ValueError("at least three validation examples are required")
    if any(value.numel() != count for value in inputs.values()):
        raise ValueError("all validation score and utility tensors must have equal length")
    if not isinstance(ridge_regularization, (int, float)) or ridge_regularization <= 0:
        raise ValueError("ridge_regularization must be positive")
    with torch.no_grad():
        design = torch.stack(
            [validation_neural_scores, validation_lm_scores, validation_length_scores], dim=1
        ).to(dtype=torch.float64)
        targets = validation_utilities.to(dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64, device=design.device)
        weights = torch.linalg.solve(design.T @ design + ridge_regularization * identity, design.T @ targets)
    return ValidationFittedScoreWeights(
        lambda_neural=float(weights[0].item()),
        lambda_lm=float(weights[1].item()),
        lambda_length=float(weights[2].item()),
        validation_examples=count,
        ridge_regularization=float(ridge_regularization),
    )


@dataclass(frozen=True)
class FactorizedScoringControl:
    """Optional explicit overrides; zero is a valid neural/LM ablation control."""

    lambda_neural: float | None = None
    lambda_lm: float | None = None
    lambda_length: float | None = None

    def __post_init__(self) -> None:
        for name, value in {
            "lambda_neural": self.lambda_neural,
            "lambda_lm": self.lambda_lm,
            "lambda_length": self.lambda_length,
        }.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{name} override must be finite or None")


@dataclass(frozen=True)
class FactorizedCandidateScoringOutput:
    neural_scores: torch.Tensor
    lm_scores: torch.Tensor
    length_scores: torch.Tensor
    combined_scores: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    effective_weights: tuple[float, float, float]

    @property
    def ranked_candidate_ids(self) -> torch.Tensor:
        return self.candidate_ids.gather(1, self.combined_scores.argsort(dim=1, descending=True))


class EvidenceFactorizedCandidateScorer(nn.Module):
    """Combine externally computed, separately auditable candidate score terms.

    The constructor requires ``ValidationFittedScoreWeights``.  Its target-free
    ``forward`` accepts only candidate score components and authorized IDs/mask;
    it cannot receive a reference text, target IDs, or validation utilities.
    """

    def __init__(
        self,
        fitted_weights: ValidationFittedScoreWeights,
        control: FactorizedScoringControl | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(fitted_weights, ValidationFittedScoreWeights):
            raise ValueError("fitted_weights must be ValidationFittedScoreWeights")
        self.fitted_weights = fitted_weights
        self.control = control or FactorizedScoringControl()

    @property
    def effective_weights(self) -> tuple[float, float, float]:
        return (
            self.fitted_weights.lambda_neural if self.control.lambda_neural is None else self.control.lambda_neural,
            self.fitted_weights.lambda_lm if self.control.lambda_lm is None else self.control.lambda_lm,
            self.fitted_weights.lambda_length if self.control.lambda_length is None else self.control.lambda_length,
        )
    def forward(
        self,
        neural_scores: torch.Tensor,
        lm_scores: torch.Tensor,
        length_scores: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> FactorizedCandidateScoringOutput:
        if neural_scores.ndim != 2 or lm_scores.shape != neural_scores.shape or length_scores.shape != neural_scores.shape:
            raise ValueError("neural_scores, lm_scores, and length_scores must be matching [batch, candidates] tensors")
        if not all(torch.is_floating_point(value) and torch.isfinite(value).all() for value in (neural_scores, lm_scores, length_scores)):
            raise ValueError("all factorized scores must be finite floating-point tensors")
        batch, candidates = neural_scores.shape
        if candidate_ids.ndim == 1:
            candidate_ids = candidate_ids.unsqueeze(0).expand(batch, -1)
        if candidate_ids.shape != (batch, candidates) or candidate_ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("candidate_ids must be integer [candidates] or [batch, candidates]")
        if candidate_ids.lt(0).any():
            raise ValueError("candidate_ids must be nonnegative")
        candidate_ids = candidate_ids.to(neural_scores.device)
        if candidate_mask is None:
            candidate_mask = torch.ones(batch, candidates, dtype=torch.bool, device=neural_scores.device)
        elif candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand(batch, -1)
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask must be [candidates] or [batch, candidates]")
        candidate_mask = candidate_mask.to(device=neural_scores.device, dtype=torch.bool)
        if not candidate_mask.any(dim=1).all():
            raise ValueError("each row needs at least one authorized candidate")
        neural_weight, lm_weight, length_weight = self.effective_weights
        combined = neural_weight * neural_scores + lm_weight * lm_scores + length_weight * length_scores
        combined = combined.masked_fill(~candidate_mask, -torch.inf)
        return FactorizedCandidateScoringOutput(
            neural_scores=neural_scores,
            lm_scores=lm_scores,
            length_scores=length_scores,
            combined_scores=combined,
            candidate_ids=candidate_ids,
            candidate_mask=candidate_mask,
            effective_weights=(neural_weight, lm_weight, length_weight),
        )
