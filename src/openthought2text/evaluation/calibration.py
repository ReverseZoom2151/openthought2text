"""Calibration and selective-prediction metrics for constrained candidate ranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    expected_calibration_error: float
    brier_score: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    coverage: float
    risk: float
    selected: int
    minimum_confidence: float


def calibration_summary(
    confidences: Sequence[float], correctness: Sequence[bool | int], *, bins: int = 10
) -> CalibrationSummary:
    """Equal-width ECE and Brier score for binary candidate correctness."""
    probabilities, outcomes = _validate(confidences, correctness)
    if bins <= 0:
        raise ValueError("bins must be positive")
    summaries: list[CalibrationBin] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [item for item, probability in enumerate(probabilities) if lower <= probability < upper or (index == bins - 1 and probability == 1.0)]
        if not members:
            summaries.append(CalibrationBin(lower, upper, 0, None, None))
            continue
        mean_confidence = sum(probabilities[item] for item in members) / len(members)
        accuracy = sum(outcomes[item] for item in members) / len(members)
        ece += len(members) / len(probabilities) * abs(mean_confidence - accuracy)
        summaries.append(CalibrationBin(lower, upper, len(members), mean_confidence, accuracy))
    brier = sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes, strict=True)) / len(probabilities)
    return CalibrationSummary(ece, brier, tuple(summaries))


def risk_coverage_curve(
    confidences: Sequence[float], correctness: Sequence[bool | int]
) -> tuple[RiskCoveragePoint, ...]:
    """Risk as low-confidence candidate predictions are progressively withheld."""
    probabilities, outcomes = _validate(confidences, correctness)
    ranked = sorted(zip(probabilities, outcomes, strict=True), key=lambda item: item[0], reverse=True)
    errors = 0
    points: list[RiskCoveragePoint] = []
    for selected, (confidence, outcome) in enumerate(ranked, start=1):
        errors += 1 - outcome
        points.append(RiskCoveragePoint(selected / len(ranked), errors / selected, selected, confidence))
    return tuple(points)


def _validate(confidences: Sequence[float], correctness: Sequence[bool | int]) -> tuple[list[float], list[int]]:
    if not confidences or len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must be non-empty and equally sized")
    probabilities = [float(item) for item in confidences]
    if any(not math.isfinite(item) or item < 0 or item > 1 for item in probabilities):
        raise ValueError("confidences must be finite probabilities in [0, 1]")
    outcomes: list[int] = []
    for item in correctness:
        if item not in (False, True, 0, 1):
            raise ValueError("correctness values must be binary")
        outcomes.append(int(item))
    return probabilities, outcomes
