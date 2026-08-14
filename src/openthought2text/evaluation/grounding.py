"""Metrics which separate neural evidence from language-prior performance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingReport:
    full_score: float
    strongest_control: str
    strongest_control_score: float
    neural_contribution: float
    grounded_gain: float
    higher_is_better: bool


def _best_control(control_scores: Mapping[str, float], higher_is_better: bool) -> tuple[str, float]:
    if not control_scores:
        raise ValueError("at least one control score is required")
    chooser = max if higher_is_better else min
    name = chooser(control_scores, key=control_scores.__getitem__)
    return name, control_scores[name]


def grounded_gain(
    full_score: float,
    control_scores: Mapping[str, float],
    *,
    higher_is_better: bool = True,
) -> tuple[float, str, float]:
    """Gain over the strongest non-neural/control baseline.

    For error metrics (CER/WER), smaller is better and a positive gain still means
    the real neural signal helped.
    """
    name, control_score = _best_control(control_scores, higher_is_better)
    gain = full_score - control_score if higher_is_better else control_score - full_score
    return gain, name, control_score


def build_grounding_report(
    full_score: float,
    control_scores: Mapping[str, float],
    *,
    shuffled_score: float,
    higher_is_better: bool = True,
) -> GroundingReport:
    gain, strongest_name, strongest_score = grounded_gain(
        full_score, control_scores, higher_is_better=higher_is_better
    )
    contribution = full_score - shuffled_score if higher_is_better else shuffled_score - full_score
    return GroundingReport(
        full_score=full_score,
        strongest_control=strongest_name,
        strongest_control_score=strongest_score,
        neural_contribution=contribution,
        grounded_gain=gain,
        higher_is_better=higher_is_better,
    )
