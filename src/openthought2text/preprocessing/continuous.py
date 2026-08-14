"""Fixed-duration preprocessing that cannot access gold text boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContinuousChunks:
    """Signal-derived chunks and their source sample boundaries."""

    values: torch.Tensor  # [chunks, channels, window_samples]
    start_samples: torch.Tensor  # [chunks]
    end_samples: torch.Tensor  # [chunks]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("values must have shape [chunks, channels, samples]")
        if self.start_samples.ndim != 1 or self.end_samples.ndim != 1:
            raise ValueError("chunk boundaries must be one-dimensional")
        if len(self.start_samples) != self.values.shape[0] or len(self.end_samples) != self.values.shape[0]:
            raise ValueError("one start/end boundary is required per chunk")
        if not torch.all(self.end_samples > self.start_samples):
            raise ValueError("every chunk end must be greater than its start")


def robust_channel_scale(signal: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Median/IQR-scale ``[channels, samples]`` signal without label information."""

    if signal.ndim != 2:
        raise ValueError("signal must have shape [channels, samples]")
    if signal.shape[-1] < 2:
        raise ValueError("at least two samples are required for robust scaling")
    median = signal.median(dim=-1, keepdim=True).values
    quantiles = torch.quantile(signal, torch.tensor([0.25, 0.75], device=signal.device), dim=-1)
    iqr = (quantiles[1] - quantiles[0]).unsqueeze(-1).clamp_min(eps)
    return (signal - median) / iqr


def chunk_continuous_signal(
    signal: torch.Tensor,
    *,
    window_samples: int,
    stride_samples: int | None = None,
    drop_last: bool = True,
) -> ContinuousChunks:
    """Chunk a recording using signal duration alone.

    This API deliberately has no target-text, word-count, or event-boundary
    arguments. It is safe for the `continuous` alignment regime when invoked on
    a recorded signal of known duration.
    """

    if signal.ndim != 2:
        raise ValueError("signal must have shape [channels, samples]")
    if window_samples < 1:
        raise ValueError("window_samples must be positive")
    stride = window_samples if stride_samples is None else stride_samples
    if stride < 1:
        raise ValueError("stride_samples must be positive")
    total_samples = signal.shape[-1]
    starts = list(range(0, total_samples - window_samples + 1, stride))
    if not starts and not drop_last:
        starts = [0]
    if not drop_last and starts and starts[-1] + window_samples < total_samples:
        starts.append(starts[-1] + stride)

    chunks: list[torch.Tensor] = []
    ends: list[int] = []
    for start in starts:
        end = min(start + window_samples, total_samples)
        value = signal[:, start:end]
        if value.shape[-1] < window_samples:
            value = torch.nn.functional.pad(value, (0, window_samples - value.shape[-1]))
        chunks.append(value)
        ends.append(end)
    if not chunks:
        empty = signal.new_empty((0, signal.shape[0], window_samples))
        bounds = torch.empty(0, dtype=torch.long, device=signal.device)
        return ContinuousChunks(empty, bounds, bounds)
    return ContinuousChunks(
        values=torch.stack(chunks),
        start_samples=torch.tensor(starts, dtype=torch.long, device=signal.device),
        end_samples=torch.tensor(ends, dtype=torch.long, device=signal.device),
    )
