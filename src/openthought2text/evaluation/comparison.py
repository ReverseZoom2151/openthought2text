"""Deterministic paired benchmark comparisons without statistical claims."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math

from .records import BenchmarkRowLabel


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class NamedBenchmarkResult:
    name: str
    label: BenchmarkRowLabel
    metrics: Mapping[str, float]
    reference_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.metrics or not self.reference_fingerprints:
            raise ValueError("name, metrics, and reference_fingerprints must be non-empty")
        if any(not str(key).strip() or not math.isfinite(float(value)) for key, value in self.metrics.items()):
            raise ValueError("metrics must be named and finite")
        if any(not str(key).strip() or not str(value).strip() for key, value in self.reference_fingerprints.items()):
            raise ValueError("reference fingerprints must be explicit")


@dataclass(frozen=True, slots=True)
class ComparisonDelta:
    metric: str
    direction: MetricDirection
    baseline_value: float
    candidate_value: float
    directional_delta: float


@dataclass(frozen=True, slots=True)
class BenchmarkComparisonArtifact:
    baseline: NamedBenchmarkResult
    candidate: NamedBenchmarkResult
    deltas: tuple[ComparisonDelta, ...]
    no_statistical_claim: str = "Deterministic point-estimate comparison only; no statistical significance claim is made."


def compare_benchmark_results(
    baseline: NamedBenchmarkResult,
    candidate: NamedBenchmarkResult,
    metric_directions: Mapping[str, MetricDirection | str],
) -> BenchmarkComparisonArtifact:
    """Compare paired systems after exact label/reference/metric contract checks."""
    if baseline.label != candidate.label:
        raise ValueError("paired comparison requires identical fully specified BenchmarkRowLabel")
    if dict(baseline.reference_fingerprints) != dict(candidate.reference_fingerprints):
        raise ValueError("paired comparison requires identical sample-to-reference fingerprints")
    metrics = set(baseline.metrics)
    if metrics != set(candidate.metrics) or metrics != set(metric_directions):
        raise ValueError("baseline, candidate, and metric_directions must contain exactly the same metrics")
    deltas: list[ComparisonDelta] = []
    for metric in sorted(metrics):
        direction = MetricDirection(metric_directions[metric])
        baseline_value, candidate_value = float(baseline.metrics[metric]), float(candidate.metrics[metric])
        delta = candidate_value - baseline_value if direction is MetricDirection.HIGHER_IS_BETTER else baseline_value - candidate_value
        deltas.append(ComparisonDelta(metric, direction, baseline_value, candidate_value, delta))
    return BenchmarkComparisonArtifact(baseline, candidate, tuple(deltas))


def render_comparison_markdown(comparison: BenchmarkComparisonArtifact) -> str:
    """Render a stable point-estimate table with an explicit non-inference disclaimer."""
    lines = ["# Paired Benchmark Comparison", "", f"Benchmark: `{comparison.baseline.label.value}`", "", f"**{comparison.no_statistical_claim}**", "", "| Metric | Direction | Baseline | Candidate | Directional delta |", "| --- | --- | ---: | ---: | ---: |"]
    for delta in comparison.deltas:
        lines.append(f"| `{delta.metric}` | {delta.direction.value} | {delta.baseline_value:.6g} | {delta.candidate_value:.6g} | {delta.directional_delta:.6g} |")
    return "\n".join(lines) + "\n"
