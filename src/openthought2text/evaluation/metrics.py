"""Dependency-light metrics used by every OpenThought2Text benchmark.

The implementations deliberately aggregate edit counts before forming a rate.  Taking
the mean of per-example rates overweights short trials, which is particularly
misleading for the small and variable-length neural-language datasets we support.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from collections import Counter
import math
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


def corpus_bleu(
    references: Iterable[str],
    hypotheses: Iterable[str],
    *,
    max_order: int = 4,
    tokenizer: callable = words,
) -> float:
    """Tokenization-explicit corpus BLEU with effective order and no smoothing.

    Effective order avoids assigning a perfect one-token corpus a zero score merely
    because it has no 2- through 4-grams. Empty hypotheses score zero; paired empty
    texts do not create artificial BLEU evidence.
    """
    refs, hyps = _paired_texts(references, hypotheses)
    if max_order <= 0:
        raise ValueError("max_order must be positive")
    matches = [0] * max_order
    totals = [0] * max_order
    reference_length = hypothesis_length = 0
    for reference, hypothesis in zip(refs, hyps, strict=True):
        ref_tokens, hyp_tokens = list(tokenizer(reference)), list(tokenizer(hypothesis))
        reference_length += len(ref_tokens)
        hypothesis_length += len(hyp_tokens)
        for order in range(1, max_order + 1):
            hyp_counts = _ngrams(hyp_tokens, order)
            ref_counts = _ngrams(ref_tokens, order)
            totals[order - 1] += sum(hyp_counts.values())
            matches[order - 1] += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
    active = [index for index, total in enumerate(totals) if total]
    if hypothesis_length == 0 or not active or any(matches[index] == 0 for index in active):
        return 0.0
    log_precision = sum(math.log(matches[index] / totals[index]) for index in active) / len(active)
    brevity_penalty = 1.0 if hypothesis_length > reference_length else math.exp(1 - reference_length / hypothesis_length)
    return brevity_penalty * math.exp(log_precision)


def corpus_rouge_l(
    references: Iterable[str], hypotheses: Iterable[str], *, tokenizer: callable = words
) -> float:
    """Macro-average token LCS F1 (ROUGE-L style), with paired-empty score one."""
    refs, hyps = _paired_texts(references, hypotheses)
    scores = [_rouge_l_tokens(list(tokenizer(reference)), list(tokenizer(hypothesis))) for reference, hypothesis in zip(refs, hyps, strict=True)]
    return sum(scores) / len(scores)


def corpus_meteor_unigram_approx(
    references: Iterable[str], hypotheses: Iterable[str], *, tokenizer: callable = words
) -> float:
    """Macro-average exact-unigram METEOR-style score, not the official METEOR.

    It uses greedy one-to-one exact matching and the common harmonic/fragmentation
    form. It deliberately omits stemming, synonym matching, and language resources.
    """
    refs, hyps = _paired_texts(references, hypotheses)
    scores = [_meteor_unigram_tokens(list(tokenizer(reference)), list(tokenizer(hypothesis))) for reference, hypothesis in zip(refs, hyps, strict=True)]
    return sum(scores) / len(scores)


def _paired_texts(references: Iterable[str], hypotheses: Iterable[str]) -> tuple[list[str], list[str]]:
    refs, hyps = list(references), list(hypotheses)
    if not refs or len(refs) != len(hyps):
        raise ValueError("references and hypotheses must be non-empty and equally sized")
    return refs, hyps


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index:index + order]) for index in range(max(0, len(tokens) - order + 1)))


def _rouge_l_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    previous = [0] * (len(hypothesis) + 1)
    for reference_token in reference:
        current = [0]
        for index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(previous[index - 1] + 1 if reference_token == hypothesis_token else max(previous[index], current[index - 1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(hypothesis), lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


def _meteor_unigram_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    used_reference: set[int] = set()
    matches: list[tuple[int, int]] = []
    for hypothesis_index, token in enumerate(hypothesis):
        match = next((index for index, reference_token in enumerate(reference) if index not in used_reference and reference_token == token), None)
        if match is not None:
            used_reference.add(match)
            matches.append((hypothesis_index, match))
    count = len(matches)
    if count == 0:
        return 0.0
    precision, recall = count / len(hypothesis), count / len(reference)
    harmonic = 10 * precision * recall / (recall + 9 * precision)
    chunks = 1 + sum(current[0] != previous[0] + 1 or current[1] != previous[1] + 1 for previous, current in zip(matches, matches[1:]))
    penalty = 0.5 * (chunks / count) ** 3
    return harmonic * (1 - penalty)


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
