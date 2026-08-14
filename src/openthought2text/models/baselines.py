"""Small target-free neural encoder baselines sharing one audited output API."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .channels import CoordinateChannelMerger
from .types import NeuralEncoderOutput, TokenTiming


def _validate_encoder_inputs(signals, sample_mask, channel_mask, sample_rate_hz):
    if signals.ndim != 3:
        raise ValueError("signals must be [batch, channels, samples]")
    batch, channels, samples = signals.shape
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if sample_mask is None:
        sample_mask = torch.ones(batch, samples, dtype=torch.bool, device=signals.device)
    if channel_mask is None:
        channel_mask = torch.ones(batch, channels, dtype=torch.bool, device=signals.device)
    if sample_mask.shape != (batch, samples):
        raise ValueError("sample_mask must be [batch, samples]")
    if channel_mask.shape != (batch, channels):
        raise ValueError("channel_mask must be [batch, channels]")
    if not channel_mask.bool().any(dim=1).all():
        raise ValueError("each example needs at least one valid channel")
    return sample_mask.bool(), channel_mask.bool()


def _downsample_mask(mask, stride_samples, output_length):
    pooled = (
        F.max_pool1d(
            mask.float().unsqueeze(1),
            kernel_size=stride_samples,
            stride=stride_samples,
            ceil_mode=True,
        )
        .squeeze(1)
        .bool()
    )
    if pooled.shape[1] < output_length:
        pooled = F.pad(pooled, (0, output_length - pooled.shape[1]), value=False)
    return pooled[:, :output_length]


def _output(features, mask, stride_samples, samples, sample_rate_hz):
    batch, tokens, _ = features.shape
    start = (
        torch.arange(tokens, device=features.device).unsqueeze(0).expand(batch, -1) * stride_samples
    )
    end = torch.clamp(start + stride_samples, max=samples)
    return NeuralEncoderOutput(
        features * mask.unsqueeze(-1).to(features.dtype),
        mask,
        TokenTiming(start, end, sample_rate_hz),
        stride_samples,
    )


class _ChannelTemporalFrontEnd(nn.Module):
    def __init__(self, hidden_size, temporal_kernel, stride_samples):
        super().__init__()
        self.hidden_size, self.stride_samples = hidden_size, stride_samples
        self.temporal = nn.Sequential(
            nn.Conv1d(
                1, hidden_size, temporal_kernel, stride=stride_samples, padding=temporal_kernel // 2
            ),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, signals, sample_mask, channel_mask):
        batch, channels, samples = signals.shape
        masked = (
            signals
            * sample_mask.unsqueeze(1).to(signals.dtype)
            * channel_mask.unsqueeze(-1).to(signals.dtype)
        )
        temporal = self.temporal(masked.reshape(batch * channels, 1, samples))
        tokens = temporal.shape[-1]
        return temporal.view(batch, channels, self.hidden_size, tokens).permute(
            0, 1, 3, 2
        ), _downsample_mask(sample_mask, self.stride_samples, tokens)


class ChannelNetNeuralEncoder(nn.Module):
    """Channelwise temporal convolution followed by coordinate-aware fusion."""

    def __init__(self, hidden_size=128, temporal_kernel=9, stride_samples=4, coordinate_size=3):
        super().__init__()
        if hidden_size < 1 or temporal_kernel < 1 or stride_samples < 1:
            raise ValueError("hidden_size, temporal_kernel, and stride_samples must be positive")
        self.hidden_size, self.stride_samples = hidden_size, stride_samples
        self.frontend = _ChannelTemporalFrontEnd(hidden_size, temporal_kernel, stride_samples)
        self.channel_merger, self.norm = (
            CoordinateChannelMerger(hidden_size, coordinate_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self, signals, sample_mask=None, channel_mask=None, coordinates=None, sample_rate_hz=200.0
    ):
        sample_mask, channel_mask = _validate_encoder_inputs(
            signals, sample_mask, channel_mask, sample_rate_hz
        )
        per_channel, token_mask = self.frontend(signals, sample_mask, channel_mask)
        return _output(
            self.norm(self.channel_merger(per_channel, channel_mask, coordinates)),
            token_mask,
            self.stride_samples,
            signals.shape[-1],
            sample_rate_hz,
        )


class GRUNeuralEncoder(nn.Module):
    """Compact GRU baseline over geometry-fused temporal channel features."""

    def __init__(
        self, hidden_size=128, temporal_kernel=9, stride_samples=4, num_layers=1, coordinate_size=3
    ):
        super().__init__()
        if hidden_size < 1 or temporal_kernel < 1 or stride_samples < 1 or num_layers < 1:
            raise ValueError(
                "hidden_size, temporal_kernel, stride_samples, and num_layers must be positive"
            )
        self.hidden_size, self.stride_samples = hidden_size, stride_samples
        self.frontend = _ChannelTemporalFrontEnd(hidden_size, temporal_kernel, stride_samples)
        self.channel_merger = CoordinateChannelMerger(hidden_size, coordinate_size)
        self.gru, self.norm = (
            nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self, signals, sample_mask=None, channel_mask=None, coordinates=None, sample_rate_hz=200.0
    ):
        sample_mask, channel_mask = _validate_encoder_inputs(
            signals, sample_mask, channel_mask, sample_rate_hz
        )
        per_channel, token_mask = self.frontend(signals, sample_mask, channel_mask)
        lengths = token_mask.sum(dim=1, dtype=torch.long)
        expected = torch.arange(token_mask.shape[1], device=token_mask.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        if not torch.equal(token_mask, expected):
            raise ValueError("GRU baseline requires a prefix-valid sample_mask after downsampling")
        merged = self.channel_merger(per_channel, channel_mask, coordinates)
        packed = nn.utils.rnn.pack_padded_sequence(
            merged, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_features, _ = self.gru(packed)
        features, _ = nn.utils.rnn.pad_packed_sequence(
            packed_features, batch_first=True, total_length=merged.shape[1]
        )
        return _output(
            self.norm(features), token_mask, self.stride_samples, signals.shape[-1], sample_rate_hz
        )


class _CompactConformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.ffn_one_norm, self.ffn_one = (
            nn.LayerNorm(hidden_size),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            ),
        )
        self.attention_norm, self.attention = (
            nn.LayerNorm(hidden_size),
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True),
        )
        self.conv_norm, self.pointwise_in = (
            nn.LayerNorm(hidden_size),
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
        )
        self.depthwise, self.pointwise_out = (
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        self.ffn_two_norm, self.ffn_two, self.output_norm = (
            nn.LayerNorm(hidden_size),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            ),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, features, token_mask):
        values = token_mask.unsqueeze(-1).to(features.dtype)
        features = (features + 0.5 * self.ffn_one(self.ffn_one_norm(features))) * values
        normed = self.attention_norm(features)
        attended, _ = self.attention(
            normed, normed, normed, key_padding_mask=~token_mask, need_weights=False
        )
        features = (features + attended) * values
        conv = self.conv_norm(features).transpose(1, 2)
        conv = self.pointwise_out(
            F.silu(self.depthwise(F.glu(self.pointwise_in(conv), dim=1)))
        ).transpose(1, 2)
        features = (features + conv) * values
        return (
            self.output_norm((features + 0.5 * self.ffn_two(self.ffn_two_norm(features))) * values)
            * values
        )


class CompactConformerNeuralEncoder(nn.Module):
    """Compact continuous Conformer baseline after coordinate-aware fusion."""

    def __init__(
        self,
        hidden_size=128,
        temporal_kernel=9,
        stride_samples=4,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        coordinate_size=3,
    ):
        super().__init__()
        if hidden_size < 1 or temporal_kernel < 1 or stride_samples < 1 or num_layers < 1:
            raise ValueError(
                "hidden_size, temporal_kernel, stride_samples, and num_layers must be positive"
            )
        if hidden_size % num_heads:
            raise ValueError("hidden_size must divide evenly by num_heads")
        self.hidden_size, self.stride_samples = hidden_size, stride_samples
        self.frontend = _ChannelTemporalFrontEnd(hidden_size, temporal_kernel, stride_samples)
        self.channel_merger = CoordinateChannelMerger(hidden_size, coordinate_size)
        self.blocks = nn.ModuleList(
            [_CompactConformerBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, signals, sample_mask=None, channel_mask=None, coordinates=None, sample_rate_hz=200.0
    ):
        sample_mask, channel_mask = _validate_encoder_inputs(
            signals, sample_mask, channel_mask, sample_rate_hz
        )
        per_channel, token_mask = self.frontend(signals, sample_mask, channel_mask)
        features = self.channel_merger(
            per_channel, channel_mask, coordinates
        ) * token_mask.unsqueeze(-1).to(signals.dtype)
        for block in self.blocks:
            features = block(features, token_mask)
        return _output(features, token_mask, self.stride_samples, signals.shape[-1], sample_rate_hz)
