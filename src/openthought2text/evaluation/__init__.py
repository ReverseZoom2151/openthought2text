"""Evaluation, grounding, and inference-audit primitives."""

from .audit import (
    LabelInvarianceResult,
    assert_label_invariance,
    assert_target_free_signature,
    audit_label_invariance,
    forbidden_generation_parameters,
)
from .grounding import GroundingReport, build_grounding_report, grounded_gain
from .metrics import (
    ErrorRate,
    RetrievalMetrics,
    character_error_rate,
    corpus_character_error_rate,
    corpus_word_error_rate,
    edit_distance,
    retrieval_metrics,
    retrieval_ranks,
    word_error_rate,
)

__all__ = [
    "ErrorRate", "GroundingReport", "LabelInvarianceResult", "RetrievalMetrics",
    "assert_label_invariance", "assert_target_free_signature", "audit_label_invariance",
    "build_grounding_report", "character_error_rate", "corpus_character_error_rate",
    "corpus_word_error_rate", "edit_distance", "forbidden_generation_parameters",
    "grounded_gain", "retrieval_metrics", "retrieval_ranks", "word_error_rate",
]
