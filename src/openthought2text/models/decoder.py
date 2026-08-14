"""Target-free autoregressive decoding from neural evidence.

The split between :meth:`forward` and :meth:`generate` is intentional.  The
former is a teacher-forced training operation; the latter cannot accept target
IDs, labels, or any text-derived conditioning.  This makes accidental
teacher-forcing during evaluation harder to write and easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderGenerationConfig:
    """Generation-only controls, deliberately excluding any target sequence."""

    max_new_tokens: int = 32
    eos_token_id: int | None = None
    temperature: float = 1.0
    do_sample: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class DecoderTrainingOutput:
    """Teacher-forced logits and optional masked next-token loss."""

    logits: torch.Tensor
    loss: torch.Tensor | None


class TargetFreeAutoregressiveDecoder(nn.Module):
    """A small cross-attention language decoder grounded in neural features.

    ``generate(neural_features, neural_mask, config)`` is the only inference
    API and takes no target argument.  It always starts from ``bos_token_id``
    and feeds back its own emitted tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_sequence_length: int = 128,
        bos_token_id: int = 0,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must divide evenly by num_heads")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if not 0 <= bos_token_id < vocab_size or not 0 <= pad_token_id < vocab_size:
            raise ValueError("BOS and PAD IDs must belong to the vocabulary")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def _validate_memory(self, neural_features: torch.Tensor, neural_mask: torch.Tensor) -> None:
        if neural_features.ndim != 3 or neural_features.shape[-1] != self.hidden_size:
            raise ValueError("neural_features must be [batch, tokens, hidden_size]")
        if neural_mask.shape != neural_features.shape[:2]:
            raise ValueError("neural_mask must be [batch, tokens]")
        if not neural_mask.bool().any(dim=1).all():
            raise ValueError("each example needs at least one valid neural token")

    def _decode(self, input_ids: torch.Tensor, neural_features: torch.Tensor, neural_mask: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        if length > self.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        target = self.token_embedding(input_ids) + self.position_embedding(positions)
        # True identifies future positions that must not be attended to.
        causal_mask = torch.triu(
            torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        decoded = self.decoder(
            tgt=target,
            memory=neural_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=input_ids.eq(self.pad_token_id),
            memory_key_padding_mask=~neural_mask.bool(),
        )
        return self.output_projection(self.output_norm(decoded))

    def forward(
        self,
        neural_features: torch.Tensor,
        neural_mask: torch.Tensor,
        target_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> DecoderTrainingOutput:
        """Teacher-forced training pass; target text is permitted only here.

        ``target_ids`` are the tokens to predict.  Decoder inputs are a BOS
        token followed by the preceding target token.  ``labels`` defaults to
        ``target_ids`` and may use ``-100`` for ignored positions.
        """
        self._validate_memory(neural_features, neural_mask)
        if target_ids.ndim != 2 or target_ids.shape[0] != neural_features.shape[0]:
            raise ValueError("target_ids must be [batch, target_tokens]")
        if target_ids.shape[1] > self.max_sequence_length:
            raise ValueError("target_ids exceeds max_sequence_length")
        if labels is not None and labels.shape != target_ids.shape:
            raise ValueError("labels must match target_ids")
        safe_targets = target_ids.masked_fill(target_ids.lt(0), self.pad_token_id)
        if safe_targets.ge(self.vocab_size).any():
            raise ValueError("target_ids contains a token outside the vocabulary")
        decoder_inputs = torch.full_like(safe_targets, self.bos_token_id)
        if target_ids.shape[1] > 1:
            decoder_inputs[:, 1:] = safe_targets[:, :-1]
        logits = self._decode(decoder_inputs, neural_features, neural_mask)
        effective_labels = target_ids if labels is None else labels
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), effective_labels.reshape(-1), ignore_index=-100)
        return DecoderTrainingOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        neural_features: torch.Tensor,
        neural_mask: torch.Tensor,
        config: DecoderGenerationConfig | None = None,
    ) -> torch.Tensor:
        """Autoregressively emit IDs using neural evidence and prior outputs only."""
        self._validate_memory(neural_features, neural_mask)
        config = config or DecoderGenerationConfig()
        if config.max_new_tokens > self.max_sequence_length:
            raise ValueError("max_new_tokens exceeds max_sequence_length")
        batch = neural_features.shape[0]
        generated = torch.full(
            (batch, 1), self.bos_token_id, dtype=torch.long, device=neural_features.device
        )
        finished = torch.zeros(batch, dtype=torch.bool, device=neural_features.device)
        for _ in range(config.max_new_tokens):
            logits = self._decode(generated, neural_features, neural_mask)[:, -1] / config.temperature
            if config.do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            else:
                next_token = logits.argmax(dim=-1)
            if config.eos_token_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, self.pad_token_id), next_token)
                finished |= next_token.eq(config.eos_token_id)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if config.eos_token_id is not None and finished.all():
                break
        return generated[:, 1:]
