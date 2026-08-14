"""Assembly and measurement utilities for timestamped continuous decoding windows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class WindowMergePolicy(str, Enum):
    CONCATENATE = "concatenate"
    DROP_LATER_OVERLAP = "drop_later_overlap"
    DROP_EARLIER_OVERLAP = "drop_earlier_overlap"


@dataclass(frozen=True, slots=True)
class TimestampedPredictionWindow:
    start_s: float
    end_s: float
    prediction_text: str
    inference_duration_s: float | None = None
    emitted_at_s: float | None = None

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.start_s)
            and math.isfinite(self.end_s)
            and 0 <= self.start_s < self.end_s
        ):
            raise ValueError("window must satisfy finite 0 <= start_s < end_s")
        if not isinstance(self.prediction_text, str) or not self.prediction_text.strip():
            raise ValueError("prediction_text must be non-empty")
        if self.inference_duration_s is not None and (
            not math.isfinite(self.inference_duration_s) or self.inference_duration_s < 0
        ):
            raise ValueError("inference_duration_s must be finite and non-negative")
        if self.emitted_at_s is not None and (
            not math.isfinite(self.emitted_at_s) or self.emitted_at_s < self.end_s
        ):
            raise ValueError("emitted_at_s must be finite and no earlier than the window end")


@dataclass(frozen=True, slots=True)
class ContinuousCoverage:
    source_duration_s: float
    assembled_duration_s: float
    overlap_duration_s: float
    gap_duration_s: float
    source_window_count: int
    assembled_window_count: int

    @property
    def fully_covered(self) -> bool:
        return self.assembled_duration_s >= self.source_duration_s - 1e-12


@dataclass(frozen=True, slots=True)
class ContinuousAssembly:
    policy: WindowMergePolicy
    selected_windows: tuple[TimestampedPredictionWindow, ...]
    merged_text: str
    coverage: ContinuousCoverage


@dataclass(frozen=True, slots=True)
class ContinuousTimingSummary:
    total_inference_s: float | None
    real_time_factor: float | None
    mean_latency_s: float | None
    max_latency_s: float | None
    timed_windows: int


def assemble_continuous_windows(
    windows: Sequence[TimestampedPredictionWindow],
    *,
    policy: WindowMergePolicy | str = WindowMergePolicy.CONCATENATE,
    max_gap_s: float | None = None,
    require_full_coverage: bool = False,
) -> ContinuousAssembly:
    """Merge windows under an explicit overlap policy and report retained coverage.

    Text is never token-trimmed from overlapping windows without token timestamps.
    Therefore ``concatenate`` preserves all evidence; drop policies discard whole
    windows and make any resulting coverage loss visible in the returned report.
    """
    if not windows:
        raise ValueError("continuous assembly requires at least one prediction window")
    if max_gap_s is not None and (not math.isfinite(max_gap_s) or max_gap_s < 0):
        raise ValueError("max_gap_s must be finite and non-negative")
    policy = WindowMergePolicy(policy)
    ordered = tuple(sorted(windows, key=lambda window: (window.start_s, window.end_s)))
    source_union, overlap, gaps = _coverage_duration(ordered)
    if max_gap_s is not None and any(gap > max_gap_s for gap in gaps):
        raise ValueError("window coverage contains a gap larger than max_gap_s")
    selected = _apply_policy(ordered, policy)
    assembled_union, _, _ = _coverage_duration(selected)
    coverage = ContinuousCoverage(
        source_union, assembled_union, overlap, sum(gaps), len(ordered), len(selected)
    )
    if require_full_coverage and not coverage.fully_covered:
        raise ValueError("selected merge policy does not retain full source coverage")
    return ContinuousAssembly(
        policy, selected, " ".join(window.prediction_text for window in selected), coverage
    )


def summarize_continuous_timing(assembly: ContinuousAssembly) -> ContinuousTimingSummary:
    """Summarize observed compute and emission timing; this makes no capability claim."""
    durations = [
        window.inference_duration_s
        for window in assembly.selected_windows
        if window.inference_duration_s is not None
    ]
    latencies = [
        window.emitted_at_s - window.end_s
        for window in assembly.selected_windows
        if window.emitted_at_s is not None
    ]
    total = None if len(durations) != len(assembly.selected_windows) else sum(durations)
    rtf = (
        None
        if total is None or assembly.coverage.source_duration_s == 0
        else total / assembly.coverage.source_duration_s
    )
    return ContinuousTimingSummary(
        total,
        rtf,
        None if not latencies else sum(latencies) / len(latencies),
        None if not latencies else max(latencies),
        len(latencies),
    )


def _apply_policy(
    windows: Sequence[TimestampedPredictionWindow], policy: WindowMergePolicy
) -> tuple[TimestampedPredictionWindow, ...]:
    if policy is WindowMergePolicy.CONCATENATE:
        return tuple(windows)
    selected: list[TimestampedPredictionWindow] = []
    for window in windows:
        if policy is WindowMergePolicy.DROP_LATER_OVERLAP:
            if any(_overlap(window, prior) > 0 for prior in selected):
                continue
            selected.append(window)
        else:
            selected = [prior for prior in selected if _overlap(window, prior) <= 0]
            selected.append(window)
    return tuple(selected)


def _coverage_duration(
    windows: Sequence[TimestampedPredictionWindow],
) -> tuple[float, float, list[float]]:
    if not windows:
        return 0.0, 0.0, []
    union = 0.0
    overlap = 0.0
    gaps: list[float] = []
    start, end = windows[0].start_s, windows[0].end_s
    for window in windows[1:]:
        if window.start_s > end:
            union += end - start
            gaps.append(window.start_s - end)
            start, end = window.start_s, window.end_s
        else:
            overlap += max(0.0, min(end, window.end_s) - window.start_s)
            end = max(end, window.end_s)
    return union + end - start, overlap, gaps


def _overlap(left: TimestampedPredictionWindow, right: TimestampedPredictionWindow) -> float:
    return max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
