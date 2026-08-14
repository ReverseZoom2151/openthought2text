"""Small subject-conditioning adapters, deliberately separate from decoding."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


SubjectAdapterMode = Literal["identity", "additive", "film"]


class SubjectAdapter(nn.Module):
    """Apply an identity, additive embedding, or FiLM subject adaptation."""

    def __init__(self, hidden_size: int, num_subjects: int, mode: SubjectAdapterMode = "identity") -> None:
        super().__init__()
        if mode not in {"identity", "additive", "film"}:
            raise ValueError("mode must be identity, additive, or film")
        self.hidden_size = hidden_size
        self.num_subjects = num_subjects
        self.mode = mode
        self.embedding: nn.Embedding | None
        if mode == "identity":
            self.embedding = None
        elif mode == "additive":
            self.embedding = nn.Embedding(num_subjects, hidden_size)
            nn.init.zeros_(self.embedding.weight)
        else:
            self.embedding = nn.Embedding(num_subjects, hidden_size * 2)
            with torch.no_grad():
                self.embedding.weight.zero_()
                self.embedding.weight[:, :hidden_size].fill_(1.0)

    def forward(self, features: torch.Tensor, subject_ids: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must be [batch, time, hidden]")
        if features.shape[-1] != self.hidden_size:
            raise ValueError("features have an unexpected hidden size")
        if self.mode == "identity":
            return features
        if subject_ids is None or subject_ids.shape != (features.shape[0],):
            raise ValueError("subject_ids must be [batch] for non-identity adapters")
        assert self.embedding is not None
        params = self.embedding(subject_ids)
        if self.mode == "additive":
            return features + params.unsqueeze(1)
        scale, shift = params.chunk(2, dim=-1)
        return features * scale.unsqueeze(1) + shift.unsqueeze(1)
