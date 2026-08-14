"""A compact continuous neural encoder with transparent masking and timing."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .channels import CoordinateChannelMerger
from .types import NeuralEncoderOutput, TokenTiming


class ContinuousNeuralEncoder(nn.Module):
    """Depthwise temporal convolutions followed by a Transformer encoder.

    Input is ``[batch, channels, samples]``.  Temporal validity is supplied by
    ``sample_mask`` and sensor availability by ``channel_mask``.  The module is
    intentionally modest: it is a reproducible baseline, not a claim that a
    particular architecture yields grounded language decoding.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        temporal_kernel: int = 9,
        stride_samples: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        coordinate_size: int = 3,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must divide evenly by num_heads")
        if temporal_kernel < 1 or stride_samples < 1:
            raise ValueError("temporal_kernel and stride_samples must be positive")
        self.hidden_size = hidden_size
        self.stride_samples = stride_samples
        padding = temporal_kernel // 2
        self.temporal = nn.Sequential(
            nn.LazyConv1d(hidden_size, temporal_kernel, stride=stride_samples, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.GELU(),
        )
        self.channel_merger = CoordinateChannelMerger(hidden_size, coordinate_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    def _downsample_mask(self, mask: torch.Tensor, output_length: int) -> torch.Tensor:
        # Max pooling records whether the receptive window contains evidence.
        pooled = (
            F.max_pool1d(
                mask.float().unsqueeze(1),
                kernel_size=self.stride_samples,
                stride=self.stride_samples,
                ceil_mode=True,
            )
            .squeeze(1)
            .bool()
        )
        if pooled.shape[1] < output_length:
            pooled = F.pad(pooled, (0, output_length - pooled.shape[1]), value=False)
        return pooled[:, :output_length]

    def forward(
        self,
        signals: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        sample_rate_hz: float = 200.0,
    ) -> NeuralEncoderOutput:
        if signals.ndim != 3:
            raise ValueError("signals must be [batch, channels, samples]")
        batch, channels, samples = signals.shape
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if sample_mask is None:
            sample_mask = torch.ones(batch, samples, dtype=torch.bool, device=signals.device)
        if sample_mask.shape != (batch, samples):
            raise ValueError("sample_mask must be [batch, samples]")
        if channel_mask is None:
            channel_mask = torch.ones(batch, channels, dtype=torch.bool, device=signals.device)
        if channel_mask.shape != (batch, channels):
            raise ValueError("channel_mask must be [batch, channels]")
        if not channel_mask.any(dim=1).all():
            raise ValueError("each sample needs at least one valid channel")

        masked = signals * sample_mask.unsqueeze(1).to(signals.dtype)
        masked = masked * channel_mask.unsqueeze(-1).to(signals.dtype)
        # Process channels independently, then merge them with their geometry.
        per_channel = self.temporal(masked.reshape(batch * channels, 1, samples))
        tokens = per_channel.shape[-1]
        per_channel = per_channel.view(batch, channels, self.hidden_size, tokens).permute(
            0, 1, 3, 2
        )
        token_mask = self._downsample_mask(sample_mask, tokens)
        merged = self.channel_merger(per_channel, channel_mask, coordinates)
        merged = merged * token_mask.unsqueeze(-1).to(merged.dtype)
        contextual = self.context(merged, src_key_padding_mask=~token_mask)
        features = self.final_norm(contextual) * token_mask.unsqueeze(-1).to(contextual.dtype)
        start = (
            torch.arange(tokens, device=signals.device).unsqueeze(0).expand(batch, -1)
            * self.stride_samples
        )
        end = torch.clamp(start + self.stride_samples, max=samples)
        return NeuralEncoderOutput(
            features=features,
            mask=token_mask,
            timing=TokenTiming(start=start, end=end, sample_rate_hz=sample_rate_hz),
            stride_samples=self.stride_samples,
        )
