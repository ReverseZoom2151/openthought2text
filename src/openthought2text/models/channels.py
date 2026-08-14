"""Sensor geometry-aware merging with exact padded-channel invariance."""

from __future__ import annotations

import torch
from torch import nn


class CoordinateChannelMerger(nn.Module):
    """Merge per-channel features while respecting unavailable electrodes.

    The merger never lets a masked channel contribute a value, coordinate, or
    attention mass.  Thus appending arbitrary padded channels leaves valid
    outputs invariant up to ordinary floating-point reduction roundoff.
    """

    def __init__(self, hidden_size: int, coordinate_size: int = 3) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.coordinate_size = coordinate_size
        self.coordinate_projection = nn.Linear(coordinate_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        channel_features: torch.Tensor,
        channel_mask: torch.Tensor,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[batch, time, hidden]`` from ``[batch, channels, time, hidden]``."""
        if channel_features.ndim != 4:
            raise ValueError("channel_features must be [batch, channels, time, hidden]")
        batch, channels, _, hidden = channel_features.shape
        if hidden != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden}")
        if channel_mask.shape != (batch, channels):
            raise ValueError("channel_mask must be [batch, channels]")
        valid = channel_mask.to(dtype=torch.bool, device=channel_features.device)
        if not valid.any(dim=1).all():
            raise ValueError("each sample needs at least one valid channel")

        values = channel_features
        if coordinates is not None:
            if coordinates.ndim == 2:
                coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1)
            if coordinates.shape != (batch, channels, self.coordinate_size):
                raise ValueError("coordinates must be [channels, xyz] or [batch, channels, xyz]")
            coord = self.coordinate_projection(coordinates.to(channel_features.dtype))
            values = values + coord.unsqueeze(2)

        # Scores are per channel/time.  Fill instead of multiplying after a
        # softmax: multiplying retains a denominator dependent on padding.
        logits = self.score(values).squeeze(-1).transpose(1, 2)  # [B, T, C]
        logits = logits.masked_fill(~valid.unsqueeze(1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        merged = torch.einsum("btc,bcth->bth", weights, values)
        return self.output_norm(merged)
