"""Channel geometry checks before graph or coordinate-merger models."""

from __future__ import annotations

import torch


def validate_channel_geometry(
    channel_names: tuple[str, ...], coordinates: torch.Tensor, channel_mask: torch.Tensor
) -> None:
    """Validate `[channels, 3]` coordinates and ensure masked channels are named."""

    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [channels, 3]")
    if channel_mask.ndim != 1 or channel_mask.shape[0] != coordinates.shape[0]:
        raise ValueError("channel_mask must have shape [channels]")
    if len(channel_names) != coordinates.shape[0]:
        raise ValueError("channel_names must match coordinate count")
    if len(set(channel_names)) != len(channel_names) or any(not name.strip() for name in channel_names):
        raise ValueError("channel names must be unique, non-empty strings")
    if not torch.isfinite(coordinates[channel_mask]).all():
        raise ValueError("coordinates for active channels must be finite")
