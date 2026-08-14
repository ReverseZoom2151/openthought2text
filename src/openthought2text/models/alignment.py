"""Grounded neural pooling and contrastive brain/text alignment primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticPoolingOutput:
    """Semantic query tokens, their mean-pooled representation, and attention."""

    query_features: torch.Tensor  # [batch, queries, hidden]
    pooled: torch.Tensor  # [batch, hidden]
    attention_weights: torch.Tensor  # [batch, heads, queries, neural_tokens]


class SemanticQueryPooler(nn.Module):
    """Let a small learned set of semantic queries attend to neural evidence.

    The supplied mask is used as a key/value padding mask, so arbitrary values
    at padded neural positions cannot change the pooled representation.
    """

    def __init__(self, hidden_size: int, num_queries: int = 4, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size < 1 or num_queries < 1 or num_heads < 1 or hidden_size % num_heads:
            raise ValueError("hidden_size must divide by num_heads and queries must be positive")
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_size))
        nn.init.normal_(self.queries, std=hidden_size**-0.5)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, neural_features: torch.Tensor, neural_mask: torch.Tensor) -> SemanticPoolingOutput:
        if neural_features.ndim != 3 or neural_features.shape[-1] != self.hidden_size:
            raise ValueError("neural_features must be [batch, tokens, hidden_size]")
        if neural_mask.shape != neural_features.shape[:2]:
            raise ValueError("neural_mask must be [batch, tokens]")
        valid = neural_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("each example needs at least one valid neural token")
        queries = self.queries.unsqueeze(0).expand(neural_features.shape[0], -1, -1)
        attended, weights = self.attention(
            query=queries,
            key=neural_features,
            value=neural_features,
            key_padding_mask=~valid,
            need_weights=True,
            average_attn_weights=False,
        )
        query_features = self.norm(attended + queries)
        return SemanticPoolingOutput(
            query_features=query_features,
            pooled=query_features.mean(dim=1),
            attention_weights=weights,
        )


@dataclass(frozen=True)
class ContrastiveAlignmentOutput:
    """Loss terms and the logit/mask audit trail for a contrastive batch."""

    loss: torch.Tensor
    neural_to_text_loss: torch.Tensor
    text_to_neural_loss: torch.Tensor
    logits: torch.Tensor
    false_negative_mask: torch.Tensor


class GroupAwareSymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE with optional same-stimulus negative exclusion.

    With ``group_ids`` supplied, distinct rows belonging to one stimulus group
    are removed from the denominator.  Their diagonal pairing remains valid;
    this prevents repeated windows/annotations of a stimulus becoming false
    negatives while retaining one positive per aligned pair.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        neural_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> ContrastiveAlignmentOutput:
        if neural_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("embeddings must be [batch, embedding_dim]")
        if neural_embeddings.shape != text_embeddings.shape:
            raise ValueError("neural and text embeddings must have identical shapes")
        batch, dimension = neural_embeddings.shape
        if batch < 1 or dimension < 1:
            raise ValueError("embeddings must be nonempty")
        if group_ids is not None and group_ids.shape != (batch,):
            raise ValueError("group_ids must be [batch]")
        neural = F.normalize(neural_embeddings, dim=-1)
        text = F.normalize(text_embeddings, dim=-1)
        logits = neural @ text.t() / self.temperature
        false_negative_mask = torch.zeros(batch, batch, dtype=torch.bool, device=logits.device)
        if group_ids is not None:
            same_group = group_ids.reshape(-1, 1).eq(group_ids.reshape(1, -1))
            false_negative_mask = same_group & ~torch.eye(batch, device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(false_negative_mask, -torch.inf)
        targets = torch.arange(batch, device=logits.device)
        neural_to_text = F.cross_entropy(logits, targets)
        text_to_neural = F.cross_entropy(logits.t(), targets)
        return ContrastiveAlignmentOutput(
            loss=(neural_to_text + text_to_neural) / 2,
            neural_to_text_loss=neural_to_text,
            text_to_neural_loss=text_to_neural,
            logits=logits,
            false_negative_mask=false_negative_mask,
        )
