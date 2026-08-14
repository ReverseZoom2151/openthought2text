"""Ordered semantic-anchor decoding with a target-free generation boundary."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class OrderedSemanticAnchorDecoderConfig:
    num_anchors: int
    hidden_size: int
    num_layers: int = 2
    num_heads: int = 4
    max_sequence_length: int = 32
    pad_anchor_id: int = 0
    eos_anchor_id: int | None = None
    dropout: float = 0.1
    def __post_init__(self):
        if self.num_anchors < 2 or self.hidden_size < 1 or self.num_layers < 1 or self.num_heads < 1 or self.max_sequence_length < 1:
            raise ValueError("anchor decoder dimensions must be positive")
        if self.hidden_size % self.num_heads: raise ValueError("hidden_size must divide evenly by num_heads")
        if not 0 <= self.pad_anchor_id < self.num_anchors: raise ValueError("pad_anchor_id must belong to the anchor vocabulary")
        if self.eos_anchor_id is not None and not 0 <= self.eos_anchor_id < self.num_anchors: raise ValueError("eos_anchor_id must belong to the anchor vocabulary")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0, 1)")

@dataclass(frozen=True)
class OrderedSemanticAnchorGenerationConfig:
    max_new_anchors: int = 16
    eos_anchor_id: int | None = None
    def __post_init__(self):
        if self.max_new_anchors < 1: raise ValueError("max_new_anchors must be positive")

@dataclass(frozen=True)
class OrderedSemanticAnchorTrainingOutput:
    logits: torch.Tensor
    loss: torch.Tensor
    target_mask: torch.Tensor

@dataclass(frozen=True)
class OrderedSemanticAnchorGenerationOutput:
    anchor_ids: torch.Tensor
    anchor_mask: torch.Tensor

class OrderedSemanticAnchorDecoder(nn.Module):
    """Cross-attention anchor decoder; only training_forward accepts targets."""
    def __init__(self, config: OrderedSemanticAnchorDecoderConfig):
        super().__init__(); self.config = config; self.start_token_id = config.num_anchors
        self.embedding = nn.Embedding(config.num_anchors + 1, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        layer = nn.TransformerDecoderLayer(config.hidden_size, config.num_heads, config.hidden_size * 4, config.dropout, "gelu", batch_first=True, norm_first=True)
        self.decoder, self.norm, self.output = nn.TransformerDecoder(layer, config.num_layers), nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.num_anchors, bias=False)
    def _memory(self, features, mask):
        if features.ndim != 3 or features.shape[-1] != self.config.hidden_size: raise ValueError("neural_features must be [batch, tokens, hidden_size]")
        if mask.shape != features.shape[:2] or not mask.bool().any(dim=1).all(): raise ValueError("neural_mask must match features and retain one token per example")
    def _decode(self, ids, features, mask):
        batch, length = ids.shape
        if length > self.config.max_sequence_length: raise ValueError("anchor sequence exceeds max_sequence_length")
        pos = torch.arange(length, device=ids.device).unsqueeze(0).expand(batch, -1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1)
        values = self.embedding(ids) + self.position_embedding(pos)
        decoded = self.decoder(values, features, tgt_mask=causal, tgt_key_padding_mask=ids.eq(self.config.pad_anchor_id), memory_key_padding_mask=~mask.bool())
        return self.output(self.norm(decoded))
    def training_forward(self, neural_features, neural_mask, anchor_targets, target_mask=None):
        self._memory(neural_features, neural_mask)
        if anchor_targets.ndim != 2 or anchor_targets.shape[0] != neural_features.shape[0] or anchor_targets.shape[1] > self.config.max_sequence_length: raise ValueError("anchor_targets must be bounded [batch, anchors]")
        if target_mask is None: target_mask = anchor_targets.ne(self.config.pad_anchor_id)
        if target_mask.shape != anchor_targets.shape: raise ValueError("target_mask must match anchor_targets")
        valid = target_mask.bool()
        if (anchor_targets[valid] < 0).any() or (anchor_targets[valid] >= self.config.num_anchors).any(): raise ValueError("valid anchor_targets must belong to the anchor vocabulary")
        safe = anchor_targets.masked_fill(~valid, self.config.pad_anchor_id)
        inputs = torch.full_like(safe, self.start_token_id)
        if inputs.shape[1] > 1: inputs[:, 1:] = safe[:, :-1]
        logits = self._decode(inputs, neural_features, neural_mask)
        labels = safe.masked_fill(~valid, -100)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100) if valid.any() else logits.sum() * 0.0
        return OrderedSemanticAnchorTrainingOutput(logits, loss, valid)
    @torch.no_grad()
    def generate(self, neural_features, neural_mask, config: OrderedSemanticAnchorGenerationConfig | None = None):
        self._memory(neural_features, neural_mask); config = config or OrderedSemanticAnchorGenerationConfig()
        if config.max_new_anchors > self.config.max_sequence_length: raise ValueError("max_new_anchors exceeds max_sequence_length")
        eos = self.config.eos_anchor_id if config.eos_anchor_id is None else config.eos_anchor_id
        if eos is not None and not 0 <= eos < self.config.num_anchors: raise ValueError("generation eos_anchor_id must belong to the anchor vocabulary")
        batch = neural_features.shape[0]; inputs = torch.full((batch, 1), self.start_token_id, dtype=torch.long, device=neural_features.device); finished = torch.zeros(batch, dtype=torch.bool, device=neural_features.device); rows=[]; masks=[]
        for _ in range(config.max_new_anchors):
            next_ids = self._decode(inputs, neural_features, neural_mask)[:, -1].argmax(dim=-1); active = ~finished
            rows.append(torch.where(active, next_ids, torch.full_like(next_ids, self.config.pad_anchor_id))); masks.append(active)
            if eos is not None: finished |= next_ids.eq(eos)
            inputs = torch.cat([inputs, next_ids.unsqueeze(1)], 1)
            if finished.all(): break
        return OrderedSemanticAnchorGenerationOutput(torch.stack(rows, 1), torch.stack(masks, 1))
