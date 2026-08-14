"""Small residual vector quantization with per-codebook health diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tokenizer import CodebookHealth, codebook_health


@dataclass(frozen=True)
class ResidualVectorQuantizerConfig:
    """Configuration for a stack of equally sized residual codebooks."""

    num_codebooks: int = 2
    codebook_size: int = 256
    embedding_dim: int = 128
    commitment_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.num_codebooks < 1 or self.codebook_size < 2 or self.embedding_dim < 1:
            raise ValueError("num_codebooks, codebook_size, and embedding_dim must be positive")
        if self.commitment_weight < 0:
            raise ValueError("commitment_weight must be nonnegative")


@dataclass(frozen=True)
class ResidualVectorQuantizerOutput:
    quantized: torch.Tensor  # [batch, tokens, embedding_dim]
    indices: torch.Tensor  # [batch, tokens, levels]
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    per_level_health: tuple[CodebookHealth, ...]

    @property
    def loss(self) -> torch.Tensor:
        return self.commitment_loss + self.codebook_loss


class ResidualVectorQuantizer(nn.Module):
    """Encode an embedding as a sum of vectors from residual codebooks.

    The input residual is re-quantized at every level.  ``encode`` and
    ``decode`` expose just the discrete representation and reconstruction,
    while ``forward`` also provides straight-through quantization and losses.
    """

    def __init__(self, config: ResidualVectorQuantizerConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = nn.ModuleList(
            [nn.Embedding(config.codebook_size, config.embedding_dim) for _ in range(config.num_codebooks)]
        )
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1 / config.codebook_size, 1 / config.codebook_size)

    def _validate_embeddings(self, embeddings: torch.Tensor, mask: torch.Tensor | None) -> None:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.embedding_dim:
            raise ValueError("embeddings must be [batch, tokens, embedding_dim]")
        if mask is not None and mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must be [batch, tokens]")

    @staticmethod
    def _nearest(residual: torch.Tensor, codebook: nn.Embedding) -> torch.Tensor:
        distances = (
            residual.square().sum(dim=-1, keepdim=True)
            - 2 * residual @ codebook.weight.t()
            + codebook.weight.square().sum(dim=-1)
        )
        return distances.argmin(dim=-1)

    def encode(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return discrete code IDs with shape ``[batch, tokens, levels]``."""
        self._validate_embeddings(embeddings, mask)
        residual = embeddings
        all_indices: list[torch.Tensor] = []
        for codebook in self.codebooks:
            indices = self._nearest(residual, codebook)
            selected = codebook(indices)
            all_indices.append(indices)
            residual = residual - selected
        return torch.stack(all_indices, dim=-1)

    def decode(self, indices: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct embeddings by summing vectors selected at each level."""
        if indices.ndim != 3 or indices.shape[-1] != self.config.num_codebooks:
            raise ValueError("indices must be [batch, tokens, num_codebooks]")
        if mask is not None and mask.shape != indices.shape[:2]:
            raise ValueError("mask must be [batch, tokens]")
        if indices.lt(0).any() or indices.ge(self.config.codebook_size).any():
            raise ValueError("indices contain a code outside the codebook")
        reconstruction = torch.zeros(
            *indices.shape[:2], self.config.embedding_dim, dtype=self.codebooks[0].weight.dtype, device=indices.device
        )
        for level, codebook in enumerate(self.codebooks):
            reconstruction = reconstruction + codebook(indices[..., level])
        if mask is not None:
            reconstruction = reconstruction * mask.unsqueeze(-1).to(reconstruction.dtype)
        return reconstruction

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> ResidualVectorQuantizerOutput:
        self._validate_embeddings(embeddings, mask)
        if mask is None:
            weights = torch.ones(embeddings.shape[:2], dtype=embeddings.dtype, device=embeddings.device)
        else:
            weights = mask.to(dtype=embeddings.dtype)
        denominator = weights.sum().clamp_min(1)
        residual = embeddings
        reconstruction = torch.zeros_like(embeddings)
        indices_by_level: list[torch.Tensor] = []
        commitment = embeddings.sum() * 0.0
        codebook_loss = embeddings.sum() * 0.0
        health: list[CodebookHealth] = []
        for codebook in self.codebooks:
            indices = self._nearest(residual, codebook)
            selected = codebook(indices)
            commitment = commitment + (
                (residual - selected.detach()).square().mean(dim=-1) * weights
            ).sum() / denominator
            codebook_loss = codebook_loss + ((residual.detach() - selected).square().mean(dim=-1) * weights).sum() / denominator
            reconstruction = reconstruction + selected
            # Detaching selected keeps codebook gradients confined to the
            # explicit codebook term rather than later residual decisions.
            residual = residual - selected.detach()
            indices_by_level.append(indices)
            health.append(codebook_health(indices, self.config.codebook_size, mask))
        quantized = embeddings + (reconstruction - embeddings).detach()
        if mask is not None:
            quantized = quantized * mask.unsqueeze(-1).to(quantized.dtype)
        return ResidualVectorQuantizerOutput(
            quantized=quantized,
            indices=torch.stack(indices_by_level, dim=-1),
            commitment_loss=self.config.commitment_weight * commitment,
            codebook_loss=codebook_loss,
            per_level_health=tuple(health),
        )
