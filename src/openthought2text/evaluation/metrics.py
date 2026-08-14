"""Dependency-light metrics used by every OpenThought2Text benchmark.

The implementations deliberately aggregate edit counts before forming a rate.  Taking
the mean of per-example rates overweights short trials, which is particularly
misleading for the small and variable-length neural-language datasets we support.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re
import unicodedata


_WORD_RE = re.compile(r"[^\w']+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Case-fold and collapse whitespace without silently deleting content."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def characters(text: str) -> list[str]:
    """Return normalized characters, including spaces between words."""
    return list(normalize_text(text))


def words(text: str) -> list[str]:
    """Tokenize text for WER using a stable, language-agnostic default."""
    normalized = _WORD_RE.sub(" ", normalize_text(text))
    return normalized.split()


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    """Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, reference_item in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_item in enumerate(hypothesis, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (reference_item != hypothesis_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    reference_units: int

    @property
    def rate(self) -> float:
        # Empty references are not a meaningful error-rate denominator.  Defining
        # exact empty-to-empty as zero makes per-batch reporting practical.
        if self.reference_units == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_units


def _aggregate_error_rate(
    references: Iterable[str], hypotheses: Iterable[str], unitizer: callable
) -> ErrorRate:
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError("references and hypotheses must have equal length")
    errors = 0
    units = 0
    for reference, hypothesis in zip(refs, hyps, strict=True):
        reference_units = unitizer(reference)
        errors += edit_distance(reference_units, unitizer(hypothesis))
        units += len(reference_units)
    return ErrorRate(errors=errors, reference_units=units)


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return _aggregate_error_rate([reference], [hypothesis], characters)


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return _aggregate_error_rate([reference], [hypothesis], words)


def corpus_character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> ErrorRate:
    return _aggregate_error_rate(references, hypotheses, characters)


def corpus_word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> ErrorRate:
    return _aggregate_error_rate(references, hypotheses, words)


@dataclass(frozen=True)
class RetrievalMetrics:
    queries: int
    mean_rank: float
    mean_reciprocal_rank: float
    recall_at: dict[int, float]


def retrieval_ranks(
    score_rows: Sequence[Sequence[float]], positive_indices: Sequence[int]
) -> list[int]:
    """Return one-based ranks; score ties receive the conservative worst rank."""
    if len(score_rows) != len(positive_indices):
        raise ValueError("one positive index is required for each score row")
    ranks: list[int] = []
    for scores, positive_index in zip(score_rows, positive_indices, strict=True):
        if not 0 <= positive_index < len(scores):
            raise ValueError("positive index is outside its score row")
        positive = scores[positive_index]
        ranks.append(1 + sum(score >= positive for score in scores) - 1)
    return ranks


def retrieval_metrics(
    score_rows: Sequence[Sequence[float]],
    positive_indices: Sequence[int],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    ranks = retrieval_ranks(score_rows, positive_indices)
    if not ranks:
        raise ValueError("retrieval metrics require at least one query")
    if any(k <= 0 for k in ks):
        raise ValueError("retrieval cutoffs must be positive")
    count = len(ranks)
    return RetrievalMetrics(
        queries=count,
        mean_rank=sum(ranks) / count,
        mean_reciprocal_rank=sum(1 / rank for rank in ranks) / count,
        recall_at={k: sum(rank <= k for rank in ranks) / count for k in sorted(set(ks))},
    )
