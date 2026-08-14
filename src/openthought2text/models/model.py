"""Composed, auditable neural-signal-to-text model boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import DecoderGenerationConfig, DecoderTrainingOutput, TargetFreeAutoregressiveDecoder
from .encoder import ContinuousNeuralEncoder
from .heads import SemanticAnchorHead, SemanticAnchorOutput
from .types import NeuralEncoderOutput, TokenTiming


@dataclass(frozen=True)
class NeuralToTextTrainingOutput:
    encoder: NeuralEncoderOutput
    decoder: DecoderTrainingOutput
    anchors: SemanticAnchorOutput | None

    @property
    def loss(self) -> torch.Tensor:
        """Combined available losses; callers can still weight terms explicitly."""
        result = self.decoder.loss
        if result is None:
            raise RuntimeError("training output has no decoder loss")
        if self.anchors is not None and self.anchors.loss is not None:
            result = result + self.anchors.loss
        return result


@dataclass(frozen=True)
class NeuralToTextGenerationOutput:
    """Generated IDs together with the exact neural evidence presented to text."""

    token_ids: torch.Tensor
    neural_features: torch.Tensor
    neural_mask: torch.Tensor
    timing: TokenTiming
    anchor_logits: torch.Tensor | None


class NeuralToTextModel(nn.Module):
    """Compose continuous neural encoding, optional anchors, and target-free text.

    In ``generate`` the supplied ``token_mask`` may be either a sample mask
    ``[batch, samples]`` (applied before encoding) or an encoder-token mask
    ``[batch, tokens]`` (intersected with the emitted encoder mask).  This
    accommodates datasets which provide validity at different stages without
    letting any text label enter the inference path.
    """

    def __init__(
        self,
        encoder: ContinuousNeuralEncoder,
        decoder: TargetFreeAutoregressiveDecoder,
        semantic_anchor_head: SemanticAnchorHead | None = None,
    ) -> None:
        super().__init__()
        if encoder.hidden_size != decoder.hidden_size:
            raise ValueError("encoder and decoder hidden sizes must match")
        if (
            semantic_anchor_head is not None
            and semantic_anchor_head.hidden_size != encoder.hidden_size
        ):
            raise ValueError("semantic anchor head hidden size must match encoder")
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_anchor_head = semantic_anchor_head

    def _encode(
        self,
        signals: torch.Tensor,
        channel_mask: torch.Tensor | None,
        coordinates: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        sample_rate_hz: float,
    ) -> NeuralEncoderOutput:
        if signals.ndim != 3:
            raise ValueError("signals must be [batch, channels, samples]")
        sample_mask = None
        post_mask = None
        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape[0] != signals.shape[0]:
                raise ValueError("token_mask must be [batch, samples] or [batch, encoder_tokens]")
            if token_mask.shape[1] == signals.shape[-1]:
                sample_mask = token_mask
            else:
                post_mask = token_mask
        encoded = self.encoder(
            signals,
            sample_mask=sample_mask,
            channel_mask=channel_mask,
            coordinates=coordinates,
            sample_rate_hz=sample_rate_hz,
        )
        if post_mask is None:
            return encoded
        if post_mask.shape != encoded.mask.shape:
            raise ValueError("encoder-token token_mask must match emitted token shape")
        combined_mask = encoded.mask & post_mask.bool()
        return NeuralEncoderOutput(
            features=encoded.features * combined_mask.unsqueeze(-1).to(encoded.features.dtype),
            mask=combined_mask,
            timing=encoded.timing,
            stride_samples=encoded.stride_samples,
        )

    def forward(
        self,
        signals: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        anchor_targets: torch.Tensor | None = None,
        anchor_position_mask: torch.Tensor | None = None,
        sample_rate_hz: float = 200.0,
    ) -> NeuralToTextTrainingOutput:
        """Teacher-forced training operation; targets appear only in this API."""
        encoded = self._encode(signals, channel_mask, coordinates, token_mask, sample_rate_hz)
        decoded = self.decoder(encoded.features, encoded.mask, target_ids, labels)
        anchors = None
        if self.semantic_anchor_head is not None:
            anchors = self.semantic_anchor_head(
                encoded.features, anchor_targets, anchor_position_mask
            )
        elif anchor_targets is not None or anchor_position_mask is not None:
            raise ValueError(
                "anchor targets were supplied but no semantic_anchor_head is configured"
            )
        return NeuralToTextTrainingOutput(encoder=encoded, decoder=decoded, anchors=anchors)

    @torch.no_grad()
    def generate(
        self,
        signals: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        config: DecoderGenerationConfig | None = None,
        sample_rate_hz: float = 200.0,
    ) -> NeuralToTextGenerationOutput:
        """Target-free inference returning text IDs and the neural evidence used."""
        encoded = self._encode(signals, channel_mask, coordinates, token_mask, sample_rate_hz)
        token_ids = self.decoder.generate(encoded.features, encoded.mask, config)
        anchor_logits = None
        if self.semantic_anchor_head is not None:
            anchor_logits = self.semantic_anchor_head(encoded.features).logits
        return NeuralToTextGenerationOutput(
            token_ids=token_ids,
            neural_features=encoded.features,
            neural_mask=encoded.mask,
            timing=encoded.timing,
            anchor_logits=anchor_logits,
        )
