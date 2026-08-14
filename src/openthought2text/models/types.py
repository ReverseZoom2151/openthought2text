"""Small, explicit value objects shared by neural model components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenTiming:
    """The source-sample interval represented by each emitted neural token.

    ``start`` and ``end`` have shape ``[batch, tokens]`` and use half-open
    sample intervals.  Keeping timing alongside features makes it possible to
    audit alignment without guessing the encoder stride downstream.
    """

    start: torch.Tensor
    end: torch.Tensor
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.start.shape != self.end.shape:
            raise ValueError("TokenTiming start and end must have identical shapes")
        if self.start.ndim != 2:
            raise ValueError("TokenTiming tensors must have shape [batch, tokens]")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def seconds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.start / self.sample_rate_hz, self.end / self.sample_rate_hz


@dataclass(frozen=True)
class NeuralEncoderOutput:
    """Features emitted by an encoder, plus their validity and provenance."""

    features: torch.Tensor  # [batch, tokens, hidden]
    mask: torch.Tensor  # [batch, tokens], True means valid
    timing: TokenTiming
    stride_samples: int

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, hidden]")
        if self.mask.shape != self.features.shape[:2]:
            raise ValueError("mask must have shape [batch, tokens]")
        if self.timing.start.shape != self.mask.shape:
            raise ValueError("timing and feature masks must have the same shape")
        if self.stride_samples < 1:
            raise ValueError("stride_samples must be at least one")
