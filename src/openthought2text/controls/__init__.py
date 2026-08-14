"""Leakage and neural-grounding control input constructors."""

from .signals import (
    gaussian_noise_like,
    length_only_signal,
    mask_only_signal,
    phase_randomized_surrogate,
    shuffle_batch,
    timing_only_signal,
    zero_signal,
)
from .protocol import ControlCondition, build_control

__all__ = [
    "ControlCondition", "build_control", "gaussian_noise_like", "length_only_signal", "mask_only_signal",
    "phase_randomized_surrogate", "shuffle_batch", "timing_only_signal", "zero_signal",
]
