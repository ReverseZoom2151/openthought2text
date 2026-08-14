"""Evidence-grounded ranking over an explicitly authorized candidate set."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .alignment import SemanticQueryPooler


@dataclass(frozen=True)
class CandidateRankingOutput:
    """Candidate scores together with the neural evidence used to produce them."""

    scores: torch.Tensor  # [batch, candidates], -inf for unavailable candidates
    candidate_ids: torch.Tensor  # [batch, candidates]
    candidate_mask: torch.Tensor  # [batch, candidates]
    neural_evidence: torch.Tensor  # [batch, candidate_embedding_dim]
    query_features: torch.Tensor  # [batch, queries, encoder_hidden]
    attention_weights: torch.Tensor  # [batch, heads, queries, neural_tokens]

    @property
    def ranked_candidate_ids(self) -> torch.Tensor:
        """Authorized IDs ordered by descending neural-evidence score."""
        return self.candidate_ids.gather(1, self.scores.argsort(dim=1, descending=True))


@dataclass(frozen=True)
class CandidateRankingTrainingOutput:
    """Training loss and the authorized positive positions it supervised."""

    loss: torch.Tensor
    positive_candidate_positions: torch.Tensor
    positive_scores: torch.Tensor
    valid_candidate_counts: torch.Tensor


class MaskedCandidateRankingLoss(nn.Module):
    """Cross-entropy over only an explicit ranker's authorized candidate set."""

    def forward(
        self,
        ranking: CandidateRankingOutput,
        positive_candidate_positions: torch.Tensor,
    ) -> CandidateRankingTrainingOutput:
        if (
            positive_candidate_positions.ndim != 1
            or positive_candidate_positions.shape[0] != ranking.scores.shape[0]
        ):
            raise ValueError("positive_candidate_positions must be [batch]")
        if positive_candidate_positions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("positive_candidate_positions must have an integer dtype")
        positions = positive_candidate_positions.to(device=ranking.scores.device, dtype=torch.long)
        candidates = ranking.scores.shape[1]
        if (positions < 0).any() or (positions >= candidates).any():
            raise ValueError("positive_candidate_positions must index the candidate set")
        positive_available = ranking.candidate_mask.gather(1, positions.unsqueeze(1)).squeeze(1)
        if not positive_available.all():
            bad_rows = (~positive_available).nonzero(as_tuple=False).squeeze(1).tolist()
            raise ValueError(f"positive candidate is unavailable for batch rows {bad_rows}")
        loss = F.cross_entropy(ranking.scores, positions)
        return CandidateRankingTrainingOutput(
            loss=loss,
            positive_candidate_positions=positions,
            positive_scores=ranking.scores.gather(1, positions.unsqueeze(1)).squeeze(1),
            valid_candidate_counts=ranking.candidate_mask.sum(dim=1, dtype=torch.long),
        )


class EvidenceGroundedCandidateRanker(nn.Module):
    """Rank only caller-authorized candidates from masked neural features.

    This is deliberately a constrained baseline: candidate strings/embeddings
    must be supplied by the caller, and ``forward`` has no target or label
    argument.  It is therefore suitable for calibration and grounding tests
    where free-form language generation would confound the neural signal.
    """

    def __init__(
        self,
        hidden_size: int,
        candidate_embedding_dim: int | None = None,
        num_queries: int = 4,
        num_heads: int = 4,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        candidate_embedding_dim = candidate_embedding_dim or hidden_size
        self.hidden_size = hidden_size
        self.candidate_embedding_dim = candidate_embedding_dim
        self.temperature = temperature
        self.pooler = SemanticQueryPooler(hidden_size, num_queries=num_queries, num_heads=num_heads)
        self.evidence_projection = nn.Linear(hidden_size, candidate_embedding_dim, bias=False)
        self.training_criterion = MaskedCandidateRankingLoss()

    @staticmethod
    def _expand_candidates(
        candidate_ids: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor | None,
        batch: int,
        embedding_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_ids.ndim == 1:
            candidate_ids = candidate_ids.unsqueeze(0).expand(batch, -1)
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != batch:
            raise ValueError("candidate_ids must be [candidates] or [batch, candidates]")
        if candidate_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("candidate_ids must have an integer dtype")
        if candidate_ids.lt(0).any():
            raise ValueError("candidate_ids must be nonnegative authorized IDs")
        candidates = candidate_ids.shape[1]
        if candidate_embeddings.ndim == 2:
            candidate_embeddings = candidate_embeddings.unsqueeze(0).expand(batch, -1, -1)
        if candidate_embeddings.shape != (batch, candidates, embedding_dim):
            raise ValueError(
                "candidate_embeddings must be [candidates, dim] or [batch, candidates, dim]"
            )
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch, candidates, dtype=torch.bool, device=candidate_ids.device
            )
        elif candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0).expand(batch, -1)
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate_mask must be [candidates] or [batch, candidates]")
        candidate_mask = candidate_mask.bool()
        if not candidate_mask.any(dim=1).all():
            raise ValueError("each example must supply at least one authorized candidate")
        return candidate_ids, candidate_embeddings, candidate_mask

    def forward(
        self,
        neural_features: torch.Tensor,
        neural_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> CandidateRankingOutput:
        """Score explicit candidate IDs solely from masked neural evidence."""
        pooled = self.pooler(neural_features, neural_mask)
        ids, embeddings, mask = self._expand_candidates(
            candidate_ids,
            candidate_embeddings,
            candidate_mask,
            batch=neural_features.shape[0],
            embedding_dim=self.candidate_embedding_dim,
        )
        evidence = self.evidence_projection(pooled.pooled)
        scores = F.normalize(evidence, dim=-1).unsqueeze(1) @ F.normalize(
            embeddings, dim=-1
        ).transpose(1, 2)
        scores = scores.squeeze(1) / self.temperature
        scores = scores.masked_fill(~mask, -torch.inf)
        return CandidateRankingOutput(
            scores=scores,
            candidate_ids=ids,
            candidate_mask=mask,
            neural_evidence=evidence,
            query_features=pooled.query_features,
            attention_weights=pooled.attention_weights,
        )

    def training_loss(
        self,
        ranking: CandidateRankingOutput,
        positive_candidate_positions: torch.Tensor,
    ) -> CandidateRankingTrainingOutput:
        """Training-only supervision; ``forward`` remains target-free inference."""
        return self.training_criterion(ranking, positive_candidate_positions)
