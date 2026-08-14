"""Named control conditions and a small dispatcher for experiment harnesses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .signals import (
    gaussian_noise_like,
    length_only_signal,
    mask_only_signal,
    phase_randomized_surrogate,
    shuffle_batch,
    timing_only_signal,
    zero_signal,
)


class ControlCondition(str, Enum):
    FULL = "full"
    ZERO = "zero"
    SHUFFLED = "shuffled"
    GAUSSIAN_NOISE = "noise"
    MASK_ONLY = "mask"
    LENGTH_ONLY = "length"
    TIMING_ONLY = "timing"
    PHASE_SURROGATE = "phase_surrogate"


def build_control(
    condition: ControlCondition | str,
    signal: Any,
    *,
    seed: int = 0,
    valid_mask: Any | None = None,
    valid_lengths: Any | None = None,
    event_indices: Any | None = None,
    channels: int = 1,
    time_steps: int | None = None,
) -> Any:
    """Construct a control tensor and fail if its declared side information is absent."""
    condition = ControlCondition(condition)
    if condition is ControlCondition.FULL:
        return signal
    if condition is ControlCondition.ZERO:
        return zero_signal(signal)
    if condition is ControlCondition.SHUFFLED:
        return shuffle_batch(signal, seed=seed)
    if condition is ControlCondition.GAUSSIAN_NOISE:
        return gaussian_noise_like(signal, seed=seed)
    if condition is ControlCondition.PHASE_SURROGATE:
        return phase_randomized_surrogate(signal, seed=seed)
    if condition is ControlCondition.MASK_ONLY:
        if valid_mask is None:
            raise ValueError("mask control requires valid_mask")
        return mask_only_signal(valid_mask, channels=channels)
    if condition is ControlCondition.LENGTH_ONLY:
        if valid_lengths is None:
            raise ValueError("length control requires valid_lengths")
        return length_only_signal(valid_lengths, channels=channels, max_length=time_steps)
    if condition is ControlCondition.TIMING_ONLY:
        if event_indices is None or time_steps is None:
            raise ValueError("timing control requires event_indices and time_steps")
        return timing_only_signal(event_indices, time_steps=time_steps, channels=channels)
    raise AssertionError(f"unhandled control condition: {condition}")
