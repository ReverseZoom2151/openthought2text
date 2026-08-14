"""Neural vector quantization primitives and explicit codebook health metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NeuralTokenizerConfig:
    codebook_size: int = 256
    embedding_dim: int = 128
    commitment_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.codebook_size < 2 or self.embedding_dim < 1 or self.commitment_weight < 0:
            raise ValueError("invalid neural tokenizer configuration")


@dataclass(frozen=True)
class CodebookHealth:
    perplexity: torch.Tensor
    active_fraction: torch.Tensor
    dead_fraction: torch.Tensor
    usage: torch.Tensor


@dataclass(frozen=True)
class NeuralTokenizerOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    health: CodebookHealth

    @property
    def loss(self) -> torch.Tensor:
        return self.commitment_loss + self.codebook_loss


def codebook_health(indices: torch.Tensor, codebook_size: int, mask: torch.Tensor | None = None) -> CodebookHealth:
    """Calculate usage statistics; mask selects valid token positions."""
    flat = indices.reshape(-1)
    if mask is not None:
        if mask.shape != indices.shape:
            raise ValueError("token mask must match indices")
        flat = flat[mask.reshape(-1).bool()]
    if flat.numel() == 0:
        usage = torch.zeros(codebook_size, device=indices.device, dtype=torch.float)
    else:
        usage = torch.bincount(flat, minlength=codebook_size).to(dtype=torch.float)
        usage = usage / usage.sum()
    nonzero = usage > 0
    entropy = -(usage[nonzero] * usage[nonzero].log()).sum()
    return CodebookHealth(
        perplexity=entropy.exp(),
        active_fraction=nonzero.float().mean(),
        dead_fraction=(~nonzero).float().mean(),
        usage=usage,
    )


class NeuralVectorQuantizer(nn.Module):
    """Straight-through VQ suitable for a small baseline or an ablation."""

    def __init__(self, config: NeuralTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.codebook = nn.Embedding(config.codebook_size, config.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1 / config.codebook_size, 1 / config.codebook_size)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> NeuralTokenizerOutput:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.embedding_dim:
            raise ValueError("embeddings must be [batch, tokens, embedding_dim]")
        if mask is not None and mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must be [batch, tokens]")
        distances = (
            embeddings.square().sum(dim=-1, keepdim=True)
            - 2 * embeddings @ self.codebook.weight.t()
            + self.codebook.weight.square().sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
        selected = self.codebook(indices)
        if mask is None:
            weights = torch.ones_like(indices, dtype=embeddings.dtype)
        else:
            weights = mask.to(dtype=embeddings.dtype)
        denom = weights.sum().clamp_min(1)
        commitment = ((embeddings - selected.detach()).square().mean(dim=-1) * weights).sum() / denom
        codebook = ((embeddings.detach() - selected).square().mean(dim=-1) * weights).sum() / denom
        quantized = embeddings + (selected - embeddings).detach()
        if mask is not None:
            quantized = quantized * mask.unsqueeze(-1).to(quantized.dtype)
        return NeuralTokenizerOutput(
            quantized=quantized,
            indices=indices,
            commitment_loss=self.config.commitment_weight * commitment,
            codebook_loss=codebook,
            health=codebook_health(indices, self.config.codebook_size, mask),
        )
