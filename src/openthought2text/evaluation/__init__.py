"""Evaluation, grounding, and inference-audit primitives."""

from .audit import (
    LabelInvarianceResult,
    assert_label_invariance,
    assert_target_free_signature,
    audit_label_invariance,
    forbidden_generation_parameters,
)
from .grounding import GroundingReport, build_grounding_report, grounded_gain
from .faithfulness import (
    DEFAULT_SIGNAL_CONTROLS,
    FaithfulnessConditionResult,
    FaithfulnessSuiteResult,
    GenerationAuditSummary,
    MetricSpec,
    run_faithfulness_suite,
)
from .evaluator import RetrievalInputs, evaluate_saved_predictions
from .occlusion import (
    OcclusionMetadata,
    OcclusionMode,
    OcclusionResult,
    OcclusionSuiteResult,
    OcclusionVariant,
    aggregate_occlusion_drops,
    occlude_channel_time,
    occlude_channels,
    occlude_time,
    run_occlusion_suite,
)
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
from .statistics import (
    BootstrapInterval,
    PairedPermutationResult,
    cluster_bootstrap_ci,
    paired_permutation_test,
    stimulus_bootstrap_ci,
    subject_bootstrap_ci,
)

__all__ = [
    "BenchmarkRowLabel", "BootstrapInterval", "ControlAggregate", "ControlResult",
    "DEFAULT_SIGNAL_CONTROLS",
    "EVALUATION_REPORT_VERSION",
    "ErrorRate", "EvaluationReport", "FaithfulnessConditionResult", "FaithfulnessSuiteResult",
    "GenerationAuditSummary", "GroundingReport", "LabelInvarianceResult", "MetricSpec",
    "OcclusionMetadata", "OcclusionMode", "OcclusionResult", "OcclusionSuiteResult", "OcclusionVariant",
    "PREDICTION_RECORD_VERSION", "PairedPermutationResult", "PredictionRecord", "RetrievalInputs",
    "RetrievalMetrics",
    "aggregate_control_results",
    "assert_label_invariance", "assert_target_free_signature", "audit_label_invariance",
    "build_grounding_report", "character_error_rate", "corpus_character_error_rate",
    "corpus_word_error_rate", "edit_distance", "forbidden_generation_parameters",
    "aggregate_occlusion_drops", "cluster_bootstrap_ci", "evaluate_saved_predictions", "grounded_gain",
    "occlude_channel_time", "occlude_channels", "occlude_time", "paired_permutation_test",
    "run_faithfulness_suite", "run_occlusion_suite",
    "read_evaluation_report", "read_prediction_jsonl", "retrieval_metrics",
    "retrieval_ranks", "stimulus_bootstrap_ci", "subject_bootstrap_ci", "word_error_rate",
    "write_evaluation_report", "write_prediction_jsonl",
]
