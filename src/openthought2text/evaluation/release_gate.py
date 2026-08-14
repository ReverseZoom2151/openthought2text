"""Machine-checkable evidence gates for release and benchmark claims."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openthought2text.controls import ControlCondition
from openthought2text.reporting.provenance import RunArtifactProvenance

from .faithfulness import GenerationAuditSummary
from .records import EvaluationReport


class GateFailureCode(str, Enum):
    NO_PREDICTIONS = "no_predictions"
    RUN_ID_MISMATCH = "run_id_mismatch"
    UNSAFE_INFORMATION_ACCESS = "unsafe_information_access"
    MISSING_GENERATION_AUDIT = "missing_generation_audit"
    TARGET_ACCEPTING_GENERATOR = "target_accepting_generator"
    LABEL_INVARIANCE_FAILURE = "label_invariance_failure"
    MISSING_DECLARED_CONTROL = "missing_declared_control"
    MISSING_REPORTED_CONTROL = "missing_reported_control"
    CONTROL_SAMPLE_MISMATCH = "control_sample_mismatch"
    MISSING_GROUNDING_METRIC = "missing_grounding_metric"
    INSUFFICIENT_GROUNDED_GAIN = "insufficient_grounded_gain"
    INSUFFICIENT_NEURAL_CONTRIBUTION = "insufficient_neural_contribution"


@dataclass(frozen=True, slots=True)
class GateFailure:
    code: GateFailureCode
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReleaseGatePolicy:
    """Minimum evidence needed before presenting a neural-decoding result."""

    primary_metric: str = "wer"
    minimum_grounded_gain: float = 0.0
    minimum_neural_contribution: float = 0.0
    required_controls: tuple[ControlCondition, ...] = (
        ControlCondition.FULL,
        ControlCondition.SHUFFLED,
        ControlCondition.ZERO,
        ControlCondition.GAUSSIAN_NOISE,
        ControlCondition.MASK_ONLY,
        ControlCondition.LENGTH_ONLY,
        ControlCondition.TIMING_ONLY,
        ControlCondition.PHASE_SURROGATE,
    )
    require_target_free_information_access: bool = True

    def __post_init__(self) -> None:
        if not self.primary_metric.strip():
            raise ValueError("primary_metric must be non-empty")
        if not self.required_controls or ControlCondition.FULL not in self.required_controls:
            raise ValueError("required_controls must include full")


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    policy: ReleaseGatePolicy
    failures: tuple[GateFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failure_codes(self) -> tuple[GateFailureCode, ...]:
        return tuple(failure.code for failure in self.failures)

    def require_pass(self) -> None:
        if not self.passed:
            detail = "; ".join(f"{item.code.value}: {item.message}" for item in self.failures)
            raise AssertionError(f"release evidence gate failed: {detail}")


def assess_release_evidence(
    evaluation: EvaluationReport,
    provenance: RunArtifactProvenance,
    *,
    generation_audit: GenerationAuditSummary | None,
    available_controls: Iterable[ControlCondition | str] | None = None,
    policy: ReleaseGatePolicy | None = None,
) -> ReleaseGateResult:
    """Assess whether saved evidence supports the claimed evaluation result.

    ``available_controls`` is the control set declared by the run configuration;
    it is checked separately from ``evaluation.control_results`` so an artifact
    cannot claim a control was configured but omit its output (or vice versa).
    """
    policy = policy or ReleaseGatePolicy()
    failures: list[GateFailure] = []
    if evaluation.prediction_count <= 0:
        failures.append(
            _failure(GateFailureCode.NO_PREDICTIONS, "evaluation report has no predictions")
        )
    if evaluation.run_id != provenance.run_id:
        failures.append(
            _failure(
                GateFailureCode.RUN_ID_MISMATCH,
                "evaluation and provenance reports refer to different run IDs",
                evaluation_run_id=evaluation.run_id,
                provenance_run_id=provenance.run_id,
            )
        )
    access = provenance.information_access
    if policy.require_target_free_information_access and (
        access.inference_target_text or access.inference_text_context
    ):
        failures.append(
            _failure(
                GateFailureCode.UNSAFE_INFORMATION_ACCESS,
                "inference contract exposes target text or text context",
                inference_target_text=access.inference_target_text,
                inference_text_context=access.inference_text_context,
            )
        )
    _check_generation_audit(failures, generation_audit)

    required = set(policy.required_controls)
    if available_controls is not None:
        declared = {ControlCondition(condition) for condition in available_controls}
        for condition in sorted(required.difference(declared), key=lambda item: item.value):
            failures.append(
                _failure(
                    GateFailureCode.MISSING_DECLARED_CONTROL,
                    f"required control is absent from run declaration: {condition.value}",
                    control=condition.value,
                )
            )
    reported = {result.condition: result for result in evaluation.control_results}
    for condition in sorted(required.difference(reported), key=lambda item: item.value):
        failures.append(
            _failure(
                GateFailureCode.MISSING_REPORTED_CONTROL,
                f"required control is absent from evaluation report: {condition.value}",
                control=condition.value,
            )
        )
    for condition, result in reported.items():
        if condition in required and result.examples != evaluation.prediction_count:
            failures.append(
                _failure(
                    GateFailureCode.CONTROL_SAMPLE_MISMATCH,
                    f"{condition.value} evaluates a different number of examples",
                    control=condition.value,
                    control_examples=result.examples,
                    prediction_count=evaluation.prediction_count,
                )
            )
    _check_grounding(failures, evaluation, policy)
    return ReleaseGateResult(policy=policy, failures=tuple(failures))


def _check_generation_audit(
    failures: list[GateFailure], audit: GenerationAuditSummary | None
) -> None:
    if audit is None:
        failures.append(
            _failure(
                GateFailureCode.MISSING_GENERATION_AUDIT,
                "no target-free generation audit was supplied",
            )
        )
        return
    if not audit.signature_is_target_free:
        failures.append(
            _failure(
                GateFailureCode.TARGET_ACCEPTING_GENERATOR,
                "generation API accepts a target-like parameter",
                forbidden_parameters=list(audit.forbidden_parameters),
            )
        )
    if audit.label_invariance is None or not audit.label_invariance.invariant:
        failures.append(
            _failure(
                GateFailureCode.LABEL_INVARIANCE_FAILURE,
                "label-invariance evidence is absent or predictions changed with labels",
                label_invariance_present=audit.label_invariance is not None,
            )
        )


def _check_grounding(
    failures: list[GateFailure], evaluation: EvaluationReport, policy: ReleaseGatePolicy
) -> None:
    report = evaluation.grounding.get(policy.primary_metric)
    if report is None:
        failures.append(
            _failure(
                GateFailureCode.MISSING_GROUNDING_METRIC,
                f"evaluation report has no grounded-gain evidence for {policy.primary_metric}",
                metric=policy.primary_metric,
            )
        )
        return
    if report.grounded_gain <= policy.minimum_grounded_gain:
        failures.append(
            _failure(
                GateFailureCode.INSUFFICIENT_GROUNDED_GAIN,
                "grounded gain does not exceed the release threshold",
                metric=policy.primary_metric,
                observed=report.grounded_gain,
                required_strictly_greater_than=policy.minimum_grounded_gain,
            )
        )
    if report.neural_contribution <= policy.minimum_neural_contribution:
        failures.append(
            _failure(
                GateFailureCode.INSUFFICIENT_NEURAL_CONTRIBUTION,
                "neural contribution does not exceed the release threshold",
                metric=policy.primary_metric,
                observed=report.neural_contribution,
                required_strictly_greater_than=policy.minimum_neural_contribution,
            )
        )


def _failure(code: GateFailureCode, message: str, **evidence: Any) -> GateFailure:
    return GateFailure(code=code, message=message, evidence=evidence)
