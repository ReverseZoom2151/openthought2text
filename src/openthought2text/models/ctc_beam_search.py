"""Target-free prefix beam search for CTC neural production scores."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


def _log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


@dataclass(frozen=True)
class CTCBeamSearchConfig:
    beam_width: int = 8
    blank_token_id: int = 0
    input_is_log_probs: bool = False
    neural_weight: float = 1.0
    language_weight: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.beam_width, bool) or not isinstance(self.beam_width, int) or self.beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if isinstance(self.blank_token_id, bool) or not isinstance(self.blank_token_id, int) or self.blank_token_id < 0:
            raise ValueError("blank_token_id must be a nonnegative integer")
        for name, value in {"neural_weight": self.neural_weight, "language_weight": self.language_weight}.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ValidationFittedLanguageScorer:
    """Explicit validation-fitted callable accepted by the decoder at inference."""

    score_fn: Callable[[tuple[int, ...]], float]
    fit_description: str
    validation_examples: int

    def __post_init__(self) -> None:
        if not callable(self.score_fn):
            raise ValueError("score_fn must be callable")
        if not isinstance(self.fit_description, str) or not self.fit_description.strip():
            raise ValueError("fit_description must be nonempty")
        if isinstance(self.validation_examples, bool) or not isinstance(self.validation_examples, int) or self.validation_examples < 1:
            raise ValueError("validation_examples must be a positive integer")

    def __call__(self, prefix: tuple[int, ...]) -> float:
        score = self.score_fn(prefix)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("validation-fitted language scorer must return a finite scalar")
        return float(score)


@dataclass(frozen=True)
class CTCBeamHypothesis:
    token_ids: tuple[int, ...]
    neural_score: float
    language_score: float | None
    combined_score: float


@dataclass(frozen=True)
class CTCBeamSearchOutput:
    hypotheses: tuple[tuple[CTCBeamHypothesis, ...], ...]
    neural_scores: torch.Tensor
    language_scores: torch.Tensor | None
    combined_scores: torch.Tensor
    hypothesis_mask: torch.Tensor

    @property
    def best_token_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row[0].token_ids for row in self.hypotheses)


class TargetFreeCTCBeamSearch:
    """Deterministic CTC prefix beam search with an optional frozen language term.

    ``decode`` has no target, reference, or transcript parameter.  The optional
    scorer receives only a candidate token prefix and must be wrapped in
    ``ValidationFittedLanguageScorer`` to make its validation provenance explicit.
    """

    def __init__(self, config: CTCBeamSearchConfig | None = None) -> None:
        self.config = config or CTCBeamSearchConfig()

    def _rank(self, prefix: tuple[int, ...], blank: float, nonblank: float, scorer: ValidationFittedLanguageScorer | None) -> tuple[float, float, float | None]:
        neural = _log_add(blank, nonblank)
        language = None if scorer is None else scorer(prefix)
        combined = self.config.neural_weight * neural
        if language is not None:
            combined += self.config.language_weight * language
        return neural, combined, language

    def _decode_one(self, log_probs: torch.Tensor, scorer: ValidationFittedLanguageScorer | None) -> tuple[CTCBeamHypothesis, ...]:
        # Mapping prefix -> (P(blank), P(nonblank)) in log space.
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
        for timestep in log_probs.tolist():
            next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
            for prefix, (prob_blank, prob_nonblank) in beams.items():
                for token, token_log_prob in enumerate(timestep):
                    if token == self.config.blank_token_id:
                        old_blank, old_nonblank = next_beams.get(prefix, (-math.inf, -math.inf))
                        next_beams[prefix] = (_log_add(old_blank, _log_add(prob_blank, prob_nonblank) + token_log_prob), old_nonblank)
                    elif prefix and token == prefix[-1]:
                        # Repeat without an intervening blank remains this prefix;
                        # a blank-to-token transition creates the doubled token.
                        old_blank, old_nonblank = next_beams.get(prefix, (-math.inf, -math.inf))
                        next_beams[prefix] = (old_blank, _log_add(old_nonblank, prob_nonblank + token_log_prob))
                        extended = prefix + (token,)
                        old_blank, old_nonblank = next_beams.get(extended, (-math.inf, -math.inf))
                        next_beams[extended] = (old_blank, _log_add(old_nonblank, prob_blank + token_log_prob))
                    else:
                        extended = prefix + (token,)
                        old_blank, old_nonblank = next_beams.get(extended, (-math.inf, -math.inf))
                        next_beams[extended] = (old_blank, _log_add(old_nonblank, _log_add(prob_blank, prob_nonblank) + token_log_prob))
            ranked = sorted(
                next_beams.items(),
                key=lambda item: (-self._rank(item[0], item[1][0], item[1][1], scorer)[1], item[0]),
            )[: self.config.beam_width]
            beams = dict(ranked)
        hypotheses = []
        for prefix, (prob_blank, prob_nonblank) in beams.items():
            neural, combined, language = self._rank(prefix, prob_blank, prob_nonblank, scorer)
            hypotheses.append(CTCBeamHypothesis(prefix, neural, language, combined))
        return tuple(sorted(hypotheses, key=lambda item: (-item.combined_score, item.token_ids)))

    @torch.no_grad()
    def decode(
        self,
        logits_or_log_probs: torch.Tensor,
        valid_lengths: torch.Tensor,
        language_scorer: ValidationFittedLanguageScorer | None = None,
    ) -> CTCBeamSearchOutput:
        if logits_or_log_probs.ndim != 3 or not torch.is_floating_point(logits_or_log_probs):
            raise ValueError("logits_or_log_probs must be [batch, time, vocabulary] floating-point")
        if not torch.isfinite(logits_or_log_probs).all():
            raise ValueError("logits_or_log_probs must be finite")
        batch, time, vocabulary = logits_or_log_probs.shape
        if not 0 <= self.config.blank_token_id < vocabulary:
            raise ValueError("blank_token_id must be in the input vocabulary")
        if valid_lengths.shape != (batch,) or valid_lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("valid_lengths must be integer [batch]")
        lengths = valid_lengths.to(dtype=torch.long, device=logits_or_log_probs.device)
        if (lengths < 1).any() or (lengths > time).any():
            raise ValueError("valid_lengths must be in [1, time]")
        if language_scorer is not None and not isinstance(language_scorer, ValidationFittedLanguageScorer):
            raise ValueError("language_scorer must be ValidationFittedLanguageScorer")
        if self.config.input_is_log_probs:
            normalized = torch.logsumexp(logits_or_log_probs, dim=-1)
            if not torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-4, rtol=1e-4):
                raise ValueError("input_is_log_probs requires normalized log probabilities")
            log_probs = logits_or_log_probs
        else:
            log_probs = torch.log_softmax(logits_or_log_probs, dim=-1)
        rows = [self._decode_one(log_probs[index, : length].cpu(), language_scorer) for index, length in enumerate(lengths.tolist())]
        widest = max(len(row) for row in rows)
        neural_scores = torch.full((batch, widest), -torch.inf, dtype=log_probs.dtype, device=log_probs.device)
        combined_scores = torch.full_like(neural_scores, -torch.inf)
        language_scores = None if language_scorer is None else torch.full_like(neural_scores, -torch.inf)
        hypothesis_mask = torch.zeros(batch, widest, dtype=torch.bool, device=log_probs.device)
        for row_index, row in enumerate(rows):
            for beam_index, hypothesis in enumerate(row):
                neural_scores[row_index, beam_index] = hypothesis.neural_score
                combined_scores[row_index, beam_index] = hypothesis.combined_score
                if language_scores is not None:
                    assert hypothesis.language_score is not None
                    language_scores[row_index, beam_index] = hypothesis.language_score
                hypothesis_mask[row_index, beam_index] = True
        return CTCBeamSearchOutput(tuple(rows), neural_scores, language_scores, combined_scores, hypothesis_mask)
