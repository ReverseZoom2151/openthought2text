"""Geometry-aware graph aggregation for variable EEG/MEG montages."""

from __future__ import annotations

import torch
from torch import nn


class GraphMontageAdapter(nn.Module):
    """Distance-weighted channel graph adapter with missing-channel support.

    Valid channels attend only to valid channels, with weights based on sensor
    coordinate distance.  Invalid output channels are explicitly zeroed, so
    appended/masked channels and their arbitrary features or coordinates cannot
    affect an available electrode's output.
    """

    def __init__(
        self,
        hidden_size: int,
        coordinate_size: int = 3,
        distance_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or coordinate_size < 1 or distance_temperature <= 0:
            raise ValueError("hidden_size/coordinate_size/temperature must be positive")
        self.hidden_size = hidden_size
        self.coordinate_size = coordinate_size
        self.distance_temperature = distance_temperature
        self.message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.residual = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def _coordinates(self, coordinates: torch.Tensor, batch: int, channels: int) -> torch.Tensor:
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1)
        if coordinates.shape != (batch, channels, self.coordinate_size):
            raise ValueError("coordinates must be [channels, coordinate_size] or [batch, channels, coordinate_size]")
        return coordinates

    def graph_weights(self, channel_mask: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Return row-normalized ``[batch, channels, channels]`` graph weights."""
        if channel_mask.ndim != 2:
            raise ValueError("channel_mask must be [batch, channels]")
        batch, channels = channel_mask.shape
        valid = channel_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("each sample needs at least one valid channel")
        coords = self._coordinates(coordinates, batch, channels)
        squared_distance = (coords.unsqueeze(2) - coords.unsqueeze(1)).square().sum(dim=-1)
        valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
        # An invalid query row is never consumed, but give it a finite self
        # edge so softmax cannot form NaNs before its final zeroing.
        diagonal = torch.eye(channels, dtype=torch.bool, device=valid.device).unsqueeze(0)
        safe_invalid_rows = ~valid.unsqueeze(2) & diagonal
        allowed = valid_pairs | safe_invalid_rows
        logits = -squared_distance / self.distance_temperature
        logits = logits.masked_fill(~allowed, -torch.inf)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        channel_features: torch.Tensor,
        channel_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt ``[batch, channels, time, hidden]`` features over the graph."""
        if channel_features.ndim != 4 or channel_features.shape[-1] != self.hidden_size:
            raise ValueError("channel_features must be [batch, channels, time, hidden_size]")
        batch, channels, _, _ = channel_features.shape
        if channel_mask.shape != (batch, channels):
            raise ValueError("channel_mask must be [batch, channels]")
        weights = self.graph_weights(channel_mask, coordinates)
        valid_features = channel_features * channel_mask[:, :, None, None].to(channel_features.dtype)
        neighbor_mean = torch.einsum("bij,bjth->bith", weights, valid_features)
        output = self.norm(self.residual(valid_features) + self.message(neighbor_mean))
        return output * channel_mask[:, :, None, None].to(output.dtype)
