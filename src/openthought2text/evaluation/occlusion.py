"""Channel/time occlusion probes for target-free neural decoding models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from .audit import assert_target_free_signature
from .faithfulness import MetricSpec


class OcclusionMode(str, Enum):
    ZERO = "zero"
    MASK = "mask"


@dataclass(frozen=True, slots=True)
class OcclusionMetadata:
    """Complete declaration of input information removed or retained."""

    mode: OcclusionMode
    axis: str
    channels: tuple[int, ...]
    start_time: int
    end_time: int
    signal_shape: tuple[int, int, int]

    @property
    def control_label(self) -> str:
        channels = ",".join(str(channel) for channel in self.channels) or "all"
        return f"occlusion/{self.mode.value}/{self.axis}/channels={channels}/time={self.start_time}:{self.end_time}"


@dataclass(frozen=True, slots=True)
class OcclusionVariant:
    """An occluded input plus an optional same-shape validity mask."""

    signal: Any
    signal_mask: Any | None
    metadata: OcclusionMetadata


@dataclass(frozen=True, slots=True)
class OcclusionResult:
    metadata: OcclusionMetadata
    scores: Mapping[str, float]
    metric_drops: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OcclusionSuiteResult:
    baseline_scores: Mapping[str, float]
    results: tuple[OcclusionResult, ...]

    @property
    def mean_metric_drops(self) -> dict[str, float]:
        return aggregate_occlusion_drops(self.results)


def occlude_channels(
    signal: Any, channels: Sequence[int], *, mode: OcclusionMode | str = OcclusionMode.ZERO
) -> OcclusionVariant:
    """Remove selected channels for the entire time range of a [B, C, T] signal."""
    batch, channel_count, time_steps = _shape(signal)
    selected = _channels(channels, channel_count)
    return _occlude(
        signal,
        mode=OcclusionMode(mode),
        axis="channel",
        channels=selected,
        start_time=0,
        end_time=time_steps,
        shape=(batch, channel_count, time_steps),
    )


def occlude_time(
    signal: Any,
    start_time: int,
    end_time: int,
    *,
    mode: OcclusionMode | str = OcclusionMode.ZERO,
) -> OcclusionVariant:
    """Remove all channels in the half-open time interval [start_time, end_time)."""
    batch, channel_count, time_steps = _shape(signal)
    _time_range(start_time, end_time, time_steps)
    return _occlude(
        signal,
        mode=OcclusionMode(mode),
        axis="time",
        channels=tuple(range(channel_count)),
        start_time=start_time,
        end_time=end_time,
        shape=(batch, channel_count, time_steps),
    )


def occlude_channel_time(
    signal: Any,
    channels: Sequence[int],
    start_time: int,
    end_time: int,
    *,
    mode: OcclusionMode | str = OcclusionMode.ZERO,
) -> OcclusionVariant:
    """Remove a channel-by-time rectangle, useful for localized evidence maps."""
    batch, channel_count, time_steps = _shape(signal)
    selected = _channels(channels, channel_count)
    _time_range(start_time, end_time, time_steps)
    return _occlude(
        signal,
        mode=OcclusionMode(mode),
        axis="channel_time",
        channels=selected,
        start_time=start_time,
        end_time=end_time,
        shape=(batch, channel_count, time_steps),
    )


def run_occlusion_suite(
    generator: Callable[..., Sequence[str] | str],
    signal: Any,
    references: Sequence[str],
    metrics: Sequence[MetricSpec],
    variants: Sequence[OcclusionVariant],
    *,
    pass_signal_mask: bool = False,
) -> OcclusionSuiteResult:
    """Run a target-free generator on explicit occlusion variants.

    Set ``pass_signal_mask`` only when the model's target-free inference API
    accepts ``(signal, signal_mask)``.  Mask mode always also zeroes removed
    values, preventing a model from recovering them if it ignores its mask.
    """
    assert_target_free_signature(generator)
    if not metrics:
        raise ValueError("at least one metric is required")
    if not variants:
        raise ValueError("at least one occlusion variant is required")
    baseline_predictions = _predictions(_generate(generator, signal, None, pass_signal_mask))
    _validate_prediction_count(baseline_predictions, references, "baseline")
    baseline_scores = _scores(metrics, baseline_predictions, references)
    results: list[OcclusionResult] = []
    for variant in variants:
        predictions = _predictions(
            _generate(generator, variant.signal, variant.signal_mask, pass_signal_mask)
        )
        _validate_prediction_count(predictions, references, variant.metadata.control_label)
        scores = _scores(metrics, predictions, references)
        drops = {
            metric.name: (
                baseline_scores[metric.name] - scores[metric.name]
                if metric.higher_is_better
                else scores[metric.name] - baseline_scores[metric.name]
            )
            for metric in metrics
        }
        results.append(OcclusionResult(variant.metadata, scores, drops))
    return OcclusionSuiteResult(baseline_scores, tuple(results))


def aggregate_occlusion_drops(results: Sequence[OcclusionResult]) -> dict[str, float]:
    """Mean signed drop per metric across explicit occlusion regions."""
    if not results:
        return {}
    metric_names = sorted({name for result in results for name in result.metric_drops})
    return {
        name: sum(result.metric_drops[name] for result in results if name in result.metric_drops)
        / sum(1 for result in results if name in result.metric_drops)
        for name in metric_names
    }


def _occlude(
    signal: Any,
    *,
    mode: OcclusionMode,
    axis: str,
    channels: tuple[int, ...],
    start_time: int,
    end_time: int,
    shape: tuple[int, int, int],
) -> OcclusionVariant:
    output = _clone(signal)
    mask = _mask_like(signal) if mode is OcclusionMode.MASK else None
    for batch in range(shape[0]):
        for channel in channels:
            for time in range(start_time, end_time):
                _set(output, batch, channel, time, 0.0)
                if mask is not None:
                    _set(mask, batch, channel, time, False)
    return OcclusionVariant(
        signal=output,
        signal_mask=mask,
        metadata=OcclusionMetadata(mode, axis, channels, start_time, end_time, shape),
    )


def _shape(signal: Any) -> tuple[int, int, int]:
    if hasattr(signal, "shape"):
        shape = tuple(int(item) for item in signal.shape)
    else:
        try:
            shape = (len(signal), len(signal[0]), len(signal[0][0]))
        except (IndexError, TypeError) as error:
            raise ValueError("signal must have non-empty shape [batch, channels, time]") from error
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise ValueError("signal must have non-empty shape [batch, channels, time]")
    return shape


def _channels(channels: Sequence[int], channel_count: int) -> tuple[int, ...]:
    selected = tuple(sorted(set(int(channel) for channel in channels)))
    if not selected or any(channel < 0 or channel >= channel_count for channel in selected):
        raise ValueError("channels must be a non-empty set of valid channel indices")
    return selected


def _time_range(start: int, end: int, time_steps: int) -> None:
    if not 0 <= start < end <= time_steps:
        raise ValueError("time range must satisfy 0 <= start < end <= signal time length")


def _clone(signal: Any) -> Any:
    if hasattr(signal, "clone"):
        return signal.clone()
    if isinstance(signal, (list, tuple)):
        return deepcopy(signal)
    if hasattr(signal, "copy"):
        return signal.copy()
    return deepcopy(signal)


def _mask_like(signal: Any) -> Any:
    module = type(signal).__module__
    if module.startswith("torch"):
        import torch

        return torch.ones_like(signal, dtype=torch.bool)
    if module.startswith("numpy"):
        import numpy as np

        return np.ones_like(signal, dtype=bool)
    batch, channels, time_steps = _shape(signal)
    return [[[True for _ in range(time_steps)] for _ in range(channels)] for _ in range(batch)]


def _set(value: Any, batch: int, channel: int, time: int, replacement: Any) -> None:
    value[batch][channel][time] = replacement


def _generate(generator: Callable[..., Any], signal: Any, mask: Any | None, pass_mask: bool) -> Any:
    return generator(signal, mask) if pass_mask else generator(signal)


def _predictions(value: Sequence[str] | str) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(str(item) for item in value)


def _validate_prediction_count(predictions: Sequence[str], references: Sequence[str], name: str) -> None:
    if len(predictions) != len(references):
        raise ValueError(f"{name} produced {len(predictions)} predictions for {len(references)} references")


def _scores(
    metrics: Sequence[MetricSpec], predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float]:
    scores = {metric.name: float(metric.evaluate(predictions, references)) for metric in metrics}
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("occlusion metrics must be finite")
    return scores
