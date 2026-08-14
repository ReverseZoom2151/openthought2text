"""Frozen training-text embeddings and deterministic group-safe hard negatives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TextEmbeddingContract:
    input_size: int
    embedding_size: int

    def __post_init__(self):
        if self.input_size < 1 or self.embedding_size < 1:
            raise ValueError("text embedding sizes must be positive")


@dataclass(frozen=True)
class TextEmbeddingProvenance:
    model_identifier: str
    model_fingerprint: str
    pretraining_overlap_label: str

    def __post_init__(self):
        if not self.model_identifier.strip() or not self.model_fingerprint.strip():
            raise ValueError("text provenance requires identifier and fingerprint")
        if self.pretraining_overlap_label not in {
            "disjoint",
            "unknown",
            "potential_overlap",
            "same_dataset",
        }:
            raise ValueError("pretraining_overlap_label must be explicit and valid")


class FrozenTextEmbeddingInterface(nn.Module):
    """Already-instantiated frozen encoder; no loader and no inference forward API."""

    def __init__(
        self,
        encoder: nn.Module,
        contract: TextEmbeddingContract,
        provenance: TextEmbeddingProvenance,
    ):
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise ValueError("encoder must be already-instantiated nn.Module")
        self.encoder, self.contract, self.provenance = encoder, contract, provenance
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()

    @torch.no_grad()
    def training_embeddings(self, text_features, text_mask):
        if text_features.ndim != 3 or text_features.shape[-1] != self.contract.input_size:
            raise ValueError("text_features must be [batch, tokens, contract input_size]")
        if text_mask.shape != text_features.shape[:2]:
            raise ValueError("text_mask must match text_features")
        valid = text_mask.bool()
        values = text_features * valid.unsqueeze(-1).to(text_features.dtype)
        try:
            output = self.encoder(values, valid)
        except TypeError as error:
            raise ValueError("text encoder must implement forward(features, mask)") from error
        if output.shape != (values.shape[0], self.contract.embedding_size):
            raise ValueError("text encoder must return [batch, contract embedding_size]")
        return output


@dataclass(frozen=True)
class HardNegativeSample:
    embeddings: torch.Tensor
    memory_indices: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor


class GroupAwareHardNegativeMemoryBank(nn.Module):
    """FIFO training bank that excludes same-stimulus groups from negatives."""

    def __init__(self, capacity: int, embedding_size: int):
        super().__init__()
        if capacity < 1 or embedding_size < 1:
            raise ValueError("memory capacity and embedding_size must be positive")
        self.capacity, self.embedding_size = capacity, embedding_size
        self.register_buffer("_embeddings", torch.empty(0, embedding_size))
        self.register_buffer("_groups", torch.empty(0, dtype=torch.long))

    @property
    def size(self):
        return self._groups.numel()

    @torch.no_grad()
    def add_training_embeddings(self, embeddings, group_ids):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_size:
            raise ValueError("embeddings must be [batch, embedding_size]")
        if group_ids.shape != (embeddings.shape[0],) or group_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("group_ids must be integer [batch]")
        if group_ids.lt(0).any():
            raise ValueError("group_ids must be nonnegative")
        values = embeddings.detach().to(self._embeddings.device)
        groups = group_ids.detach().to(self._groups.device, dtype=torch.long)
        self._embeddings = torch.cat([self._embeddings, values], 0)[-self.capacity :]
        self._groups = torch.cat([self._groups, groups], 0)[-self.capacity :]

    def sample_hard_negatives(self, queries, query_group_ids, k: int) -> HardNegativeSample:
        if queries.ndim != 2 or queries.shape[1] != self.embedding_size:
            raise ValueError("queries must be [batch, embedding_size]")
        if query_group_ids.shape != (queries.shape[0],) or k < 1:
            raise ValueError("query_group_ids must be [batch] and k positive")
        batch = queries.shape[0]
        out = torch.zeros(batch, k, self.embedding_size, device=queries.device)
        indices = torch.full((batch, k), -1, dtype=torch.long, device=queries.device)
        scores = torch.full((batch, k), -torch.inf, device=queries.device)
        mask = torch.zeros(batch, k, dtype=torch.bool, device=queries.device)
        bank = (
            F.normalize(self._embeddings.to(queries.device), dim=-1)
            if self.size
            else self._embeddings.to(queries.device)
        )
        q = F.normalize(queries, dim=-1)
        for row in range(batch):
            allowed = (
                (self._groups.to(queries.device) != query_group_ids[row])
                if self.size
                else torch.zeros(0, dtype=torch.bool, device=queries.device)
            )
            candidates = [
                (float((q[row] * bank[i]).sum().item()), i)
                for i in allowed.nonzero(as_tuple=False).flatten().tolist()
            ]
            for col, (score, index) in enumerate(
                sorted(candidates, key=lambda x: (-x[0], x[1]))[:k]
            ):
                out[row, col] = self._embeddings[index].to(queries.device)
                indices[row, col] = index
                scores[row, col] = score
                mask[row, col] = True
        return HardNegativeSample(out, indices, scores, mask)
