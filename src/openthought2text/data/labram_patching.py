"""Input-geometry-only patching contract for LaBraM-style signal consumers.

This module specifies deterministic tensor geometry only.  It neither imports
LaBraM checkpoints nor asserts compatibility with any external implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import torch

from .prepared import TensorBackedSample

LABRAM_INPUT_PATCH_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class LaBraMInputPatchConfig:
    """Declared sample rate and temporal patch width for input geometry only."""

    sample_rate_hz: float
    patch_size_samples: int
    version: str = LABRAM_INPUT_PATCH_VERSION

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.patch_size_samples < 1:
            raise ValueError("patch_size_samples must be positive")
        if self.version != LABRAM_INPUT_PATCH_VERSION:
            raise ValueError(f"unsupported input patch version: {self.version!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "openthought2text.labram_input_geometry",
            "version": self.version,
            "sample_rate_hz": self.sample_rate_hz,
            "patch_size_samples": self.patch_size_samples,
            "compatibility_scope": "input_geometry_only",
            "padding_policy": "zero_values_with_explicit_masks",
        }

    @property
    def checksum(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def manifest_description(self) -> dict[str, object]:
        """Metadata suitable for a canonical manifest's transform declaration."""
        return {**self.to_dict(), "checksum": self.checksum}


@dataclass(frozen=True, slots=True)
class PatchedNeuralInput:
    """Padded ``[patches, channels, patch_time]`` values with explicit masks."""

    patches: torch.Tensor
    channel_mask: torch.Tensor
    patch_time_mask: torch.Tensor
    patch_mask: torch.Tensor
    sample_id: str
    transform_checksum: str

    def __post_init__(self) -> None:
        if self.patches.ndim != 3:
            raise ValueError("patches must have shape [patches, channels, patch_time]")
        patch_count, channels, patch_size = self.patches.shape
        if self.channel_mask.dtype != torch.bool or self.channel_mask.shape != (channels,):
            raise ValueError("channel_mask must be bool with shape [channels]")
        if self.patch_time_mask.dtype != torch.bool or self.patch_time_mask.shape != (
            patch_count,
            patch_size,
        ):
            raise ValueError("patch_time_mask must be bool with shape [patches, patch_time]")
        if self.patch_mask.dtype != torch.bool or self.patch_mask.shape != (patch_count,):
            raise ValueError("patch_mask must be bool with shape [patches]")
        expected = self.channel_mask.unsqueeze(0).unsqueeze(-1) & self.patch_time_mask.unsqueeze(1)
        if torch.any(self.patches.masked_select(~expected) != 0):
            raise ValueError("patch values outside masks must be zero")
        if not torch.equal(self.patch_mask, self.patch_time_mask.any(dim=1)):
            raise ValueError("patch_mask must equal patch_time_mask.any(-1)")


def patch_tensor_backed_sample(
    row: TensorBackedSample, config: LaBraMInputPatchConfig
) -> PatchedNeuralInput:
    """Segment one prepared signal deterministically without inspecting labels."""
    if row.sample.signal.sampling_rate_hz != config.sample_rate_hz:
        raise ValueError(
            f"sample {row.sample.sample_id} sampling rate {row.sample.signal.sampling_rate_hz} "
            f"does not match patch config {config.sample_rate_hz}"
        )
    channels, time = row.values.shape
    size = config.patch_size_samples
    patch_count = (time + size - 1) // size
    padded_time = patch_count * size
    values = torch.zeros((channels, padded_time), dtype=row.values.dtype, device=row.values.device)
    time_mask = torch.zeros(padded_time, dtype=torch.bool, device=row.values.device)
    valid = row.resolved_channel_mask.unsqueeze(-1) & row.resolved_time_mask.unsqueeze(0)
    values[:, :time] = torch.where(valid, row.values, torch.zeros_like(row.values))
    time_mask[:time] = row.resolved_time_mask
    patches = values.reshape(channels, patch_count, size).permute(1, 0, 2).contiguous()
    patch_time_mask = time_mask.reshape(patch_count, size)
    return PatchedNeuralInput(
        patches=patches,
        channel_mask=row.resolved_channel_mask,
        patch_time_mask=patch_time_mask,
        patch_mask=patch_time_mask.any(dim=1),
        sample_id=row.sample.sample_id,
        transform_checksum=config.checksum,
    )
