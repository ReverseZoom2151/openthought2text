"""Explicit, provenance-preserving reduced-channel selection for prepared tensors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from .prepared import TensorBackedSample
from .schema import SignalReference


@dataclass(frozen=True, slots=True)
class NamedMontage:
    """Ordered channel contract with a declared coordinate for each channel."""

    name: str
    channel_names: tuple[str, ...]
    coordinates: Mapping[str, tuple[float, float, float]]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.channel_names:
            raise ValueError("montage name and channel_names must be non-empty")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("montage channel names must be unique and ordered")
        if any(not isinstance(name, str) or not name.strip() for name in self.channel_names):
            raise ValueError("montage channel names must be non-empty strings")
        if set(self.coordinates) != set(self.channel_names):
            raise ValueError("montage coordinates must map exactly the declared channel names")
        for name in self.channel_names:
            coordinate = self.coordinates[name]
            if len(coordinate) != 3 or any(not math.isfinite(float(value)) for value in coordinate):
                raise ValueError(
                    f"montage coordinate for {name!r} must contain three finite values"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "channel_names": list(self.channel_names),
            "coordinates": {name: list(self.coordinates[name]) for name in self.channel_names},
        }


@dataclass(frozen=True, slots=True)
class MontageProvenance:
    montage: NamedMontage
    source_sample_ids: tuple[str, ...]
    source_channel_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "montage": self.montage.to_dict(),
            "source_sample_ids": list(self.source_sample_ids),
            "source_channel_counts": list(self.source_channel_counts),
        }


def select_named_montage(
    samples: Sequence[TensorBackedSample], montage: NamedMontage
) -> tuple[tuple[TensorBackedSample, ...], MontageProvenance]:
    """Select exactly the requested named channels, in montage order.

    Samples with unnamed, duplicate, or unavailable channels are rejected rather
    than silently reordered.  Channel/time masks are subset without inspecting
    padded values.
    """
    if not samples:
        raise ValueError("cannot select a montage from no samples")
    selected: list[TensorBackedSample] = []
    original_counts: list[int] = []
    for row in samples:
        names = row.sample.signal.channel_names
        if not names:
            raise ValueError(f"sample {row.sample.sample_id} has no declared channel names")
        if len(set(names)) != len(names):
            raise ValueError(f"sample {row.sample.sample_id} has duplicate channel names")
        index_by_name = {name: index for index, name in enumerate(names)}
        missing = [name for name in montage.channel_names if name not in index_by_name]
        if missing:
            raise ValueError(
                f"sample {row.sample.sample_id} lacks montage channels: {', '.join(missing)}"
            )
        indices = torch.tensor(
            [index_by_name[name] for name in montage.channel_names], dtype=torch.long
        )
        signal = SignalReference(
            uri=row.sample.signal.uri,
            recording_id=row.sample.signal.recording_id,
            sampling_rate_hz=row.sample.signal.sampling_rate_hz,
            channel_count=len(montage.channel_names),
            array_key=row.sample.signal.array_key,
            checksum_sha256=row.sample.signal.checksum_sha256,
            channel_names=montage.channel_names,
        )
        sample = replace(
            row.sample,
            signal=signal,
            metadata={**row.sample.metadata, "montage": montage.name},
        )
        selected.append(
            TensorBackedSample(
                sample=sample,
                values=row.values.index_select(0, indices.to(row.values.device)),
                channel_mask=row.resolved_channel_mask.index_select(
                    0, indices.to(row.values.device)
                ),
                time_mask=row.resolved_time_mask,
            )
        )
        original_counts.append(row.values.shape[0])
    provenance = MontageProvenance(
        montage=montage,
        source_sample_ids=tuple(row.sample.sample_id for row in samples),
        source_channel_counts=tuple(original_counts),
    )
    return tuple(selected), provenance
