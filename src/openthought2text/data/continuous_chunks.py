"""Fixed-duration views derived solely from neural-signal timeline metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from .prepared import TensorBackedSample
from .schema import TimeInterval

SIGNAL_TIMELINE_ALIGNMENT = "signal_timeline_only"


@dataclass(frozen=True, slots=True)
class ContinuousChunkProvenance:
    """Records the target-independent policy used to create every chunk."""

    source_sample_ids: tuple[str, ...]
    duration_s: float
    stride_s: float
    alignment_regime: str
    boundary_source: str = "signal_timestamps"

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.stride_s <= 0:
            raise ValueError("continuous chunk duration and stride must be positive")
        if self.alignment_regime != SIGNAL_TIMELINE_ALIGNMENT:
            raise ValueError("continuous chunks require signal_timeline_only alignment")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sample_ids": list(self.source_sample_ids),
            "duration_s": self.duration_s,
            "stride_s": self.stride_s,
            "alignment_regime": self.alignment_regime,
            "boundary_source": self.boundary_source,
        }


@dataclass(frozen=True, slots=True)
class ContinuousChunkView:
    chunks: tuple[TensorBackedSample, ...]
    provenance: ContinuousChunkProvenance


def _sample_count(seconds: float, sampling_rate_hz: float, name: str) -> int:
    count = seconds * sampling_rate_hz
    rounded = round(count)
    if rounded < 1 or not math.isclose(count, rounded, abs_tol=1e-8):
        raise ValueError(f"{name} must resolve to a positive whole number of signal samples")
    return rounded


def build_continuous_chunk_view(
    samples: Sequence[TensorBackedSample],
    *,
    duration_s: float,
    stride_s: float,
    alignment_regime: str = SIGNAL_TIMELINE_ALIGNMENT,
) -> ContinuousChunkView:
    """Build fixed-length chunks from sample-rate timestamps only.

    The source target, token timing, text length, task, and arbitrary metadata
    are never consulted.  The final partial time window is retained with its
    explicit time mask, making chunk length fixed without inventing observations.
    Chunks deliberately carry no text target.
    """
    if not samples:
        raise ValueError("continuous chunk view needs at least one tensor-backed sample")
    provenance = ContinuousChunkProvenance(
        source_sample_ids=tuple(item.sample.sample_id for item in samples),
        duration_s=duration_s,
        stride_s=stride_s,
        alignment_regime=alignment_regime,
    )
    if len(set(provenance.source_sample_ids)) != len(provenance.source_sample_ids):
        raise ValueError("continuous chunk source sample IDs must be unique")
    chunks: list[TensorBackedSample] = []
    for row in samples:
        rate = row.sample.signal.sampling_rate_hz
        duration_samples = _sample_count(duration_s, rate, "duration_s")
        stride_samples = _sample_count(stride_s, rate, "stride_s")
        time_mask = row.resolved_time_mask
        # A chunk exists iff its timestamped window contains at least one valid
        # signal sample.  This makes masks independent of all target fields.
        for start_index in range(0, row.values.shape[1], stride_samples):
            stop_index = min(start_index + duration_samples, row.values.shape[1])
            source_mask = time_mask[start_index:stop_index]
            if not source_mask.any():
                continue
            values = torch.zeros(
                (row.values.shape[0], duration_samples),
                dtype=row.values.dtype,
                device=row.values.device,
            )
            chunk_mask = torch.zeros(duration_samples, dtype=torch.bool, device=row.values.device)
            width = stop_index - start_index
            values[:, :width] = row.values[:, start_index:stop_index]
            chunk_mask[:width] = source_mask
            valid = row.resolved_channel_mask.unsqueeze(-1) & chunk_mask.unsqueeze(0)
            values = torch.where(valid, values, torch.zeros_like(values))
            start_s = row.sample.interval.start_s + start_index / rate
            interval = TimeInterval(start_s, start_s + duration_s)
            chunk_sample = replace(
                row.sample,
                sample_id=f"{row.sample.sample_id}:chunk:{start_index}",
                interval=interval,
                target=None,
                metadata={
                    **row.sample.metadata,
                    "alignment_regime": SIGNAL_TIMELINE_ALIGNMENT,
                    "chunk_start_sample": start_index,
                    "chunk_duration_samples": duration_samples,
                    "chunk_boundary_source": "signal_timestamps",
                },
            )
            chunks.append(
                TensorBackedSample(
                    sample=chunk_sample,
                    values=values,
                    channel_mask=row.resolved_channel_mask,
                    time_mask=chunk_mask,
                )
            )
    if not chunks:
        raise ValueError("signal timestamps contain no valid samples for continuous chunks")
    return ContinuousChunkView(tuple(chunks), provenance)
