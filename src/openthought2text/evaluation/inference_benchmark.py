"""Reproducible wall-clock measurement for target-free inference callables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
import time
from typing import Any

from .audit import assert_target_free_signature


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkResult:
    warmup_count: int
    measured_count: int
    samples_per_input: int
    elapsed_wall_s: float
    samples_per_second: float
    latency_p50_s: float
    latency_p95_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def benchmark_target_free_inference(
    generator: Callable[[Any], Any],
    input_factory: Callable[[int], Any],
    *,
    warmup_count: int = 10,
    measured_count: int = 100,
    samples_per_input: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    metadata: Mapping[str, Any] | None = None,
) -> InferenceBenchmarkResult:
    """Measure a generator called strictly as ``generator(input_factory(index))``.

    Timing covers the target-free generator call, not input construction or model
    loading. Results are measurements under caller-supplied conditions, not a
    claim of general throughput, latency, or real-time capability.
    """
    assert_target_free_signature(generator)
    if warmup_count < 0 or measured_count <= 0 or samples_per_input <= 0:
        raise ValueError("warmup_count must be non-negative; measured_count and samples_per_input positive")
    for index in range(warmup_count):
        generator(input_factory(index))
    latencies: list[float] = []
    for index in range(measured_count):
        neural_input = input_factory(warmup_count + index)
        start = float(clock())
        generator(neural_input)
        elapsed = float(clock()) - start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("clock must return finite non-decreasing values")
        latencies.append(elapsed)
    elapsed_wall = sum(latencies)
    samples = measured_count * samples_per_input
    details = {"inference_path": "target_free", "input_construction": "explicit_factory", **dict(metadata or {})}
    return InferenceBenchmarkResult(
        warmup_count, measured_count, samples_per_input, elapsed_wall,
        math.inf if elapsed_wall == 0 else samples / elapsed_wall,
        _quantile(latencies, 0.5), _quantile(latencies, 0.95), details,
    )


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (position - low) * (ordered[high] - ordered[low])
