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
from .records import (
    EVALUATION_REPORT_VERSION,
    PREDICTION_RECORD_VERSION,
    BenchmarkRowLabel,
    ControlAggregate,
    ControlResult,
    EvaluationReport,
    PredictionRecord,
    aggregate_control_results,
    read_evaluation_report,
    read_prediction_jsonl,
    write_evaluation_report,
    write_prediction_jsonl,
)

__all__ = [
    "BenchmarkRowLabel", "ControlAggregate", "ControlResult", "EVALUATION_REPORT_VERSION",
    "ErrorRate", "EvaluationReport", "GroundingReport", "LabelInvarianceResult",
    "PREDICTION_RECORD_VERSION", "PredictionRecord", "RetrievalMetrics",
    "aggregate_control_results",
    "assert_label_invariance", "assert_target_free_signature", "audit_label_invariance",
    "build_grounding_report", "character_error_rate", "corpus_character_error_rate",
    "corpus_word_error_rate", "edit_distance", "forbidden_generation_parameters",
    "grounded_gain", "read_evaluation_report", "read_prediction_jsonl", "retrieval_metrics",
    "retrieval_ranks", "word_error_rate", "write_evaluation_report", "write_prediction_jsonl",
]
