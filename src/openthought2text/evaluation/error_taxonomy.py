"""Deterministic, word-level error taxonomy for saved neural-text predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from .metrics import words


class TextErrorCategory(str, Enum):
    EXACT = "exact"
    REPETITION = "repetition"
    OMISSION = "omission"
    SUBSTITUTION = "substitution"
    HALLUCINATION = "hallucination"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class EditOperations:
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0

    @property
    def total(self) -> int:
        return self.insertions + self.deletions + self.substitutions


@dataclass(frozen=True, slots=True)
class TextErrorRecord:
    sample_id: str
    reference: str
    hypothesis: str
    category: TextErrorCategory
    operations: EditOperations


@dataclass(frozen=True, slots=True)
class TextErrorReport:
    records: tuple[TextErrorRecord, ...]
    counts: dict[TextErrorCategory, int]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def rates(self) -> dict[TextErrorCategory, float]:
        if not self.records:
            return {category: 0.0 for category in TextErrorCategory}
        return {category: count / len(self.records) for category, count in self.counts.items()}


def classify_text_error(reference: str, hypothesis: str) -> tuple[TextErrorCategory, EditOperations]:
    """Classify one prediction with a documented, stable precedence order.

    Precedence is exact, empty output, repetition, pure/no-overlap hallucination,
    then the dominant word edit operation (ties resolve to substitution).  This
    maps mixed errors to one category while retaining the complete operation
    vector, so summary counts never hide the underlying lexical changes.
    """
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    operations = word_edit_operations(reference_words, hypothesis_words)
    if reference_words == hypothesis_words and hypothesis_words:
        return TextErrorCategory.EXACT, operations
    if not hypothesis_words:
        return TextErrorCategory.EMPTY, operations
    if not reference_words:
        return TextErrorCategory.HALLUCINATION, operations
    if not set(reference_words).intersection(hypothesis_words):
        return TextErrorCategory.HALLUCINATION, operations
    if _has_excess_repetition(reference_words, hypothesis_words):
        return TextErrorCategory.REPETITION, operations
    if operations.deletions and not operations.insertions and not operations.substitutions:
        return TextErrorCategory.OMISSION, operations
    if operations.insertions and not operations.deletions and not operations.substitutions:
        return TextErrorCategory.HALLUCINATION, operations
    if operations.substitutions and not operations.insertions and not operations.deletions:
        return TextErrorCategory.SUBSTITUTION, operations
    if operations.insertions > max(operations.deletions, operations.substitutions):
        return TextErrorCategory.HALLUCINATION, operations
    if operations.deletions > max(operations.insertions, operations.substitutions):
        return TextErrorCategory.OMISSION, operations
    return TextErrorCategory.SUBSTITUTION, operations


def classify_text_errors(
    references: Iterable[str], hypotheses: Iterable[str], *, sample_ids: Sequence[str] | None = None
) -> TextErrorReport:
    """Produce per-sample records and counts covering every taxonomy category."""
    reference_list = list(references)
    hypothesis_list = list(hypotheses)
    if len(reference_list) != len(hypothesis_list):
        raise ValueError("references and hypotheses must have equal length")
    if sample_ids is None:
        ids = [str(index) for index in range(len(reference_list))]
    else:
        ids = list(sample_ids)
        if len(ids) != len(reference_list) or len(ids) != len(set(ids)):
            raise ValueError("sample_ids must be unique and match the number of predictions")
    records_list: list[TextErrorRecord] = []
    for sample_id, reference, hypothesis in zip(ids, reference_list, hypothesis_list, strict=True):
        category, operations = classify_text_error(reference, hypothesis)
        records_list.append(TextErrorRecord(sample_id, reference, hypothesis, category, operations))
    records = tuple(records_list)
    counts = {category: 0 for category in TextErrorCategory}
    for record in records:
        counts[record.category] += 1
    return TextErrorReport(records=records, counts=counts)


def word_edit_operations(reference: Sequence[str], hypothesis: Sequence[str]) -> EditOperations:
    """Return a deterministic Levenshtein operation count at word level."""
    rows = len(reference)
    columns = len(hypothesis)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        costs[row][0] = row
    for column in range(columns + 1):
        costs[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if reference[row - 1] == hypothesis[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
            else:
                costs[row][column] = 1 + min(
                    costs[row - 1][column], costs[row][column - 1], costs[row - 1][column - 1]
                )
    insertions = deletions = substitutions = 0
    row, column = rows, columns
    while row or column:
        if row and column and reference[row - 1] == hypothesis[column - 1]:
            row, column = row - 1, column - 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            substitutions += 1
            row, column = row - 1, column - 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return EditOperations(insertions=insertions, deletions=deletions, substitutions=substitutions)


def _has_excess_repetition(reference: Sequence[str], hypothesis: Sequence[str]) -> bool:
    reference_counts = Counter(reference)
    hypothesis_counts = Counter(hypothesis)
    return any(
        count > 1 and count > reference_counts[word] for word, count in hypothesis_counts.items()
    )
