"""Auxiliary semantic-anchor and CTC production heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticAnchorOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class SemanticAnchorHead(nn.Module):
    """Classify a semantic anchor at selected neural-token positions.

    A position mask contributes only to the loss, rather than silently
    modifying logits.  This keeps prediction inspection possible while making
    supervision selection explicit and testable.
    """

    def __init__(self, hidden_size: int, num_anchors: int) -> None:
        super().__init__()
        if hidden_size < 1 or num_anchors < 2:
            raise ValueError("hidden_size must be positive and num_anchors at least two")
        self.hidden_size = hidden_size
        self.num_anchors = num_anchors
        self.projection = nn.Linear(hidden_size, num_anchors)

    def forward(
        self,
        features: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ) -> SemanticAnchorOutput:
        if features.ndim != 3 or features.shape[-1] != self.hidden_size:
            raise ValueError("features must be [batch, tokens, hidden_size]")
        logits = self.projection(features)
        if target_ids is None:
            if position_mask is not None:
                raise ValueError("position_mask requires target_ids")
            return SemanticAnchorOutput(logits=logits, loss=None)
        if target_ids.shape != features.shape[:2]:
            raise ValueError("target_ids must be [batch, tokens]")
        if position_mask is not None and position_mask.shape != target_ids.shape:
            raise ValueError("position_mask must match target_ids")
        labels = target_ids.clone()
        if position_mask is not None:
            labels = labels.masked_fill(~position_mask.bool(), -100)
        valid = labels.ne(-100)
        if valid.any():
            if labels[valid].lt(0).any() or labels[valid].ge(self.num_anchors).any():
                raise ValueError("anchor targets must be in range or -100")
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_anchors), labels.reshape(-1), ignore_index=-100
            )
        else:
            # A differentiable zero makes all-masked minibatches safe in a
            # multi-task training step.
            loss = logits.sum() * 0.0
        return SemanticAnchorOutput(logits=logits, loss=loss)


@dataclass(frozen=True)
class CTCProductionOutput:
    logits: torch.Tensor  # [batch, time, vocabulary]
    log_probs: torch.Tensor  # [time, batch, vocabulary]
    input_lengths: torch.Tensor  # [batch]
    loss: torch.Tensor | None


class CTCProductionHead(nn.Module):
    """Linear CTC head with prefix-mask and target-length validation."""

    def __init__(self, hidden_size: int, vocabulary_size: int, blank_token_id: int = 0) -> None:
        super().__init__()
        if hidden_size < 1 or vocabulary_size < 2:
            raise ValueError("hidden_size must be positive and vocabulary_size at least two")
        if not 0 <= blank_token_id < vocabulary_size:
            raise ValueError("blank_token_id must be in the vocabulary")
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.blank_token_id = blank_token_id
        self.projection = nn.Linear(hidden_size, vocabulary_size)
        self.ctc_loss = nn.CTCLoss(blank=blank_token_id, reduction="mean", zero_infinity=True)

    @staticmethod
    def _input_lengths(token_mask: torch.Tensor) -> torch.Tensor:
        if token_mask.ndim != 2:
            raise ValueError("token_mask must be [batch, time]")
        mask = token_mask.bool()
        lengths = mask.sum(dim=1, dtype=torch.long)
        expected = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(
            1
        )
        if not torch.equal(mask, expected):
            raise ValueError("CTC token_mask must contain one valid prefix, not holes")
        if (lengths < 1).any():
            raise ValueError("each CTC example needs at least one valid token")
        return lengths

    def forward(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> CTCProductionOutput:
        if features.ndim != 3 or features.shape[-1] != self.hidden_size:
            raise ValueError("features must be [batch, time, hidden_size]")
        if token_mask.shape != features.shape[:2]:
            raise ValueError("token_mask must be [batch, time]")
        input_lengths = self._input_lengths(token_mask)
        logits = self.projection(features)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        if (targets is None) != (target_lengths is None):
            raise ValueError("targets and target_lengths must be supplied together")
        if targets is None:
            return CTCProductionOutput(logits, log_probs, input_lengths, None)
        if target_lengths is None or target_lengths.shape != (features.shape[0],):
            raise ValueError("target_lengths must be [batch]")
        target_lengths = target_lengths.to(device=features.device, dtype=torch.long)
        if (target_lengths < 0).any() or (target_lengths > input_lengths).any():
            raise ValueError(
                "CTC target lengths must be nonnegative and no greater than input lengths"
            )
        if targets.ndim == 2:
            if targets.shape[0] != features.shape[0] or (target_lengths > targets.shape[1]).any():
                raise ValueError("padded targets have incompatible batch or length")
            packed_targets = torch.cat(
                [targets[i, : target_lengths[i]] for i in range(features.shape[0])]
            )
        elif targets.ndim == 1:
            if targets.numel() != int(target_lengths.sum().item()):
                raise ValueError("flattened targets must match the sum of target_lengths")
            packed_targets = targets
        else:
            raise ValueError("targets must be [batch, target_time] or flattened")
        packed_targets = packed_targets.to(device=features.device, dtype=torch.long)
        if packed_targets.numel() and (
            packed_targets.lt(0).any()
            or packed_targets.ge(self.vocabulary_size).any()
            or packed_targets.eq(self.blank_token_id).any()
        ):
            raise ValueError("CTC targets must be non-blank vocabulary IDs")
        loss = self.ctc_loss(log_probs, packed_targets, input_lengths, target_lengths)
        return CTCProductionOutput(logits, log_probs, input_lengths, loss)


def greedy_ctc_decode(
    logits: torch.Tensor,
    blank_token_id: int = 0,
    token_mask: torch.Tensor | None = None,
) -> list[list[int]]:
    """Argmax, collapse repeats, and remove blank IDs for each valid prefix."""
    if logits.ndim != 3:
        raise ValueError("logits must be [batch, time, vocabulary]")
    batch, time, vocabulary = logits.shape
    if not 0 <= blank_token_id < vocabulary:
        raise ValueError("blank_token_id must be in logits vocabulary")
    if token_mask is None:
        token_mask = torch.ones(batch, time, dtype=torch.bool, device=logits.device)
    if token_mask.shape != (batch, time):
        raise ValueError("token_mask must be [batch, time]")
    lengths = CTCProductionHead._input_lengths(token_mask)
    best = logits.argmax(dim=-1)
    decoded: list[list[int]] = []
    for row, length in zip(best, lengths.tolist()):
        collapsed: list[int] = []
        previous = blank_token_id
        for token in row[:length].tolist():
            if token != blank_token_id and token != previous:
                collapsed.append(token)
            previous = token
        decoded.append(collapsed)
    return decoded
