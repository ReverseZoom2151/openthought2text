"""Cluster-aware uncertainty and paired significance utilities.

Neural-language datasets typically contain many correlated windows per subject or
stimulus.  These routines resample the declared independent unit, not individual
windows, and paired tests preserve the exact pairing between full-signal and
control predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import math
import random


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    clusters: int
    resamples: int
    seed: int
    cluster_unit: str


@dataclass(frozen=True, slots=True)
class PairedPermutationResult:
    observed_difference: float
    p_value: float
    alternative: str
    pairs: int
    permutations: int
    exact: bool
    seed: int


def subject_bootstrap_ci(
    scores_by_subject: Mapping[str, Sequence[float]],
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap an equally weighted mean over subjects."""
    return cluster_bootstrap_ci(
        scores_by_subject,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        cluster_unit="subject",
    )


def stimulus_bootstrap_ci(
    scores_by_stimulus: Mapping[str, Sequence[float]],
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap an equally weighted mean over stimuli/text items."""
    return cluster_bootstrap_ci(
        scores_by_stimulus,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        cluster_unit="stimulus",
    )


def cluster_bootstrap_ci(
    scores_by_cluster: Mapping[str, Sequence[float]],
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
    cluster_unit: str = "cluster",
) -> BootstrapInterval:
    """Percentile CI for an equally weighted cluster mean.

    Each cluster is sampled with replacement and its observed mean is retained.
    This avoids falsely treating many overlapping windows from one participant as
    independent observations.  It intentionally does not estimate a new
    within-cluster distribution; use a hierarchical bootstrap only when the
    experimental design identifies both independent levels in advance.
    """
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if not scores_by_cluster:
        raise ValueError("at least one cluster is required")
    cluster_means: list[float] = []
    for cluster, values in scores_by_cluster.items():
        if not str(cluster).strip():
            raise ValueError("cluster identifiers must be non-empty")
        if not values:
            raise ValueError("each cluster must contain at least one score")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("scores must be finite")
        cluster_means.append(sum(numeric) / len(numeric))

    estimate = _mean(cluster_means)
    rng = random.Random(seed)
    samples = sorted(
        _mean([cluster_means[rng.randrange(len(cluster_means))] for _ in cluster_means])
        for _ in range(resamples)
    )
    alpha = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=estimate,
        lower=_quantile(samples, alpha),
        upper=_quantile(samples, 1 - alpha),
        confidence=confidence,
        clusters=len(cluster_means),
        resamples=resamples,
        seed=seed,
        cluster_unit=cluster_unit,
    )


def paired_permutation_test(
    full_scores: Sequence[float],
    control_scores: Sequence[float],
    *,
    higher_is_better: bool = True,
    alternative: str = "greater",
    permutations: int = 10_000,
    exact_max_pairs: int = 16,
    seed: int = 0,
) -> PairedPermutationResult:
    """Sign-flip permutation test for matched full-neural versus control scores.

    The reported difference is always positive when full neural input improves the
    metric: ``full - control`` for utility metrics, and ``control - full`` for
    error metrics.  For few pairs all sign assignments are enumerated exactly;
    otherwise deterministic Monte Carlo draws use the plus-one correction.
    """
    if len(full_scores) != len(control_scores) or not full_scores:
        raise ValueError("full_scores and control_scores must be non-empty and equally sized")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    if permutations <= 0 or exact_max_pairs < 0:
        raise ValueError("permutations must be positive and exact_max_pairs non-negative")
    full = [float(score) for score in full_scores]
    control = [float(score) for score in control_scores]
    if not all(math.isfinite(score) for score in full + control):
        raise ValueError("scores must be finite")
    differences = [
        full_score - control_score if higher_is_better else control_score - full_score
        for full_score, control_score in zip(full, control, strict=True)
    ]
    observed = _mean(differences)
    exact = len(differences) <= exact_max_pairs
    if exact:
        null_statistics = [
            _mean([sign * difference for sign, difference in zip(signs, differences, strict=True)])
            for signs in product((-1, 1), repeat=len(differences))
        ]
    else:
        rng = random.Random(seed)
        null_statistics = [
            _mean([rng.choice((-1, 1)) * difference for difference in differences])
            for _ in range(permutations)
        ]
    if alternative == "greater":
        exceedances = sum(statistic >= observed for statistic in null_statistics)
    else:
        exceedances = sum(abs(statistic) >= abs(observed) for statistic in null_statistics)
    denominator = len(null_statistics)
    p_value = exceedances / denominator if exact else (exceedances + 1) / (denominator + 1)
    return PairedPermutationResult(
        observed_difference=observed,
        p_value=p_value,
        alternative=alternative,
        pairs=len(differences),
        permutations=denominator,
        exact=exact,
        seed=seed,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated quantile with explicit endpoint behavior."""
    if not values:
        raise ValueError("quantile requires values")
    if probability <= 0:
        return values[0]
    if probability >= 1:
        return values[-1]
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    fraction = index - lower
    return values[lower] + fraction * (values[upper] - values[lower])
