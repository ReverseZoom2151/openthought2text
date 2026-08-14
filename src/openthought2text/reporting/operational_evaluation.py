"""Operational, non-claiming artifacts for failure review and control preflight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from openthought2text.controls import ControlCondition
from openthought2text.evaluation.error_taxonomy import TextErrorRecord
from openthought2text.evaluation.records import ControlResult, PredictionRecord

from .execution_spec import TargetFreeEvaluationSpec
from .provenance import ProvenanceError


OPERATIONAL_EVALUATION_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class FailureCase:
    sample_id: str
    reference: str
    full_prediction: str
    error_category: str
    control_predictions: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "reference": self.reference, "full_prediction": self.full_prediction, "error_category": self.error_category, "control_predictions": dict(sorted(self.control_predictions.items()))}


@dataclass(frozen=True, slots=True)
class FailureCaseExplorerArtifact:
    run_id: str
    prediction_artifact: str
    evaluation_artifact: str
    provenance_binding_sha256: str
    control_conditions: tuple[str, ...]
    cases: tuple[FailureCase, ...]
    no_performance_claim: str = "Failure-review artifact only; no performance claim is made."
    schema_version: str = OPERATIONAL_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.prediction_artifact.strip() or not self.evaluation_artifact.strip(): raise ProvenanceError("failure explorer requires explicit run and artifact references")
        if len({case.sample_id for case in self.cases}) != len(self.cases): raise ProvenanceError("failure explorer sample IDs must be unique")
        if self.no_performance_claim != "Failure-review artifact only; no performance claim is made.": raise ProvenanceError("failure explorer cannot contain performance claims")

    @property
    def binding_sha256(self) -> str: return _digest(self.binding_dict())
    def binding_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "run_id": self.run_id, "prediction_artifact": self.prediction_artifact, "evaluation_artifact": self.evaluation_artifact, "provenance_binding_sha256": self.provenance_binding_sha256, "control_conditions": list(self.control_conditions), "cases": [case.to_dict() for case in self.cases], "no_performance_claim": self.no_performance_claim}
    def to_dict(self) -> dict[str, Any]: return {**self.binding_dict(), "binding_sha256": self.binding_sha256}


def build_failure_case_explorer(
    records: Sequence[PredictionRecord], errors: Sequence[TextErrorRecord], control_results: Sequence[ControlResult], *, prediction_artifact: str, evaluation_artifact: str, provenance_binding_sha256: str
) -> FailureCaseExplorerArtifact:
    """Join saved full/control predictions to deterministic per-sample error records."""
    if not records or not errors: raise ValueError("saved prediction and error records are required")
    run_ids = {record.run_id for record in records}
    if len(run_ids) != 1: raise ValueError("prediction records must have one run_id")
    by_control: dict[str, dict[str, PredictionRecord]] = {}
    for record in records:
        by_control.setdefault(record.control.value, {})[record.sample_id] = record
    if ControlCondition.FULL.value not in by_control: raise ValueError("failure explorer requires full predictions")
    error_by_id = {error.sample_id: error for error in errors}
    full_ids = set(by_control[ControlCondition.FULL.value])
    if full_ids != set(error_by_id): raise ValueError("error records must match full prediction sample IDs")
    for control, rows in by_control.items():
        if set(rows) != full_ids: raise ValueError(f"{control} predictions are not paired to full samples")
    declared = {item.condition.value for item in control_results}
    if not set(by_control).issubset(declared): raise ValueError("every saved prediction control must appear in control results")
    cases = tuple(FailureCase(sample_id, error_by_id[sample_id].reference, by_control["full"][sample_id].prediction_text, error_by_id[sample_id].category.value, {control: rows[sample_id].prediction_text for control, rows in by_control.items() if control != "full"}) for sample_id in sorted(full_ids))
    return FailureCaseExplorerArtifact(next(iter(run_ids)), prediction_artifact, evaluation_artifact, provenance_binding_sha256, tuple(sorted(by_control)), cases)


def render_failure_case_explorer_markdown(artifact: FailureCaseExplorerArtifact) -> str:
    lines = ["# Failure-case explorer", "", f"**{artifact.no_performance_claim}**", "", f"Run: `{artifact.run_id}`  ", f"Binding: `{artifact.binding_sha256}`", "", "| Sample | Category | Reference | Full prediction | Control predictions |", "| --- | --- | --- | --- | --- |"]
    for case in artifact.cases:
        controls = "; ".join(f"{name}={value}" for name, value in sorted(case.control_predictions.items())) or "—"
        lines.append(f"| {case.sample_id} | {case.error_category} | {case.reference} | {case.full_prediction} | {controls} |")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ControlSuitePlanValidation:
    execution_spec_binding_sha256: str
    missing_controls: tuple[str, ...]
    missing_output_artifacts: tuple[str, ...]
    valid: bool
    no_performance_claim: str = "Preflight validation only; no performance claim is made."

    def to_dict(self) -> dict[str, Any]: return {"execution_spec_binding_sha256": self.execution_spec_binding_sha256, "missing_controls": list(self.missing_controls), "missing_output_artifacts": list(self.missing_output_artifacts), "valid": self.valid, "no_performance_claim": self.no_performance_claim}


def validate_complete_control_suite_plan(spec: TargetFreeEvaluationSpec, *, required_controls: Sequence[ControlCondition] = tuple(ControlCondition), required_outputs: Sequence[str] = ("predictions.jsonl", "evaluation.json", "provenance.json")) -> ControlSuitePlanValidation:
    """Validate that a target-free execution spec commits to the complete control suite."""
    missing_controls = tuple(sorted((item.value for item in set(required_controls).difference(spec.control_conditions))))
    declared_outputs = set(spec.required_output_artifacts)
    missing_outputs = tuple(sorted(item for item in required_outputs if item not in declared_outputs))
    return ControlSuitePlanValidation(spec.binding_sha256, missing_controls, missing_outputs, not missing_controls and not missing_outputs)


def render_control_suite_plan_markdown(validation: ControlSuitePlanValidation) -> str:
    status = "PASS" if validation.valid else "FAIL"
    return f"# Complete control-suite plan validation\n\n**{status} — {validation.no_performance_claim}**\n\nExecution-spec binding: `{validation.execution_spec_binding_sha256}`\n\n- Missing controls: {', '.join(validation.missing_controls) or 'none'}\n- Missing outputs: {', '.join(validation.missing_output_artifacts) or 'none'}\n"


def _digest(value: Any) -> str: return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
