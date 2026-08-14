"""Safe, file-backed failure-review reporting for already-saved artifacts.

This module is deliberately limited to deserializing JSON/JSONL and rendering
review material.  It never imports model code, opens checkpoints, or executes
inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openthought2text.evaluation.error_taxonomy import (
    EditOperations,
    TextErrorCategory,
    TextErrorRecord,
)
from openthought2text.evaluation.records import (
    ControlResult,
    EvaluationReport,
    PredictionRecord,
    read_prediction_jsonl,
)

from .operational_evaluation import (
    FailureCaseExplorerArtifact,
    build_failure_case_explorer,
    render_failure_case_explorer_markdown,
)
from .provenance import ProvenanceError, RunArtifactProvenance, read_provenance_report
from .visualizations import VisualizationFragment, render_failure_case_gallery


@dataclass(frozen=True, slots=True)
class FailureArtifactRender:
    """Validated failure explorer plus deterministic text and HTML-safe views."""

    explorer: FailureCaseExplorerArtifact
    explorer_markdown: str
    gallery: VisualizationFragment

    def to_markdown(self) -> str:
        return self.explorer_markdown + "\n" + self.gallery.markdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "explorer": self.explorer.to_dict(),
            "explorer_markdown": self.explorer_markdown,
            "gallery": {"markdown": self.gallery.markdown, "html": self.gallery.html},
            "no_performance_claim": "Saved-artifact review only; no evaluation was executed.",
        }


def build_failure_artifact_render(
    prediction_payload: Sequence[Mapping[str, Any]],
    error_payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    control_payload: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    prediction_artifact: str,
    evaluation_artifact: str,
    provenance: Mapping[str, Any],
) -> FailureArtifactRender:
    """Purely validate saved payloads, then build review-only failure views."""
    records = tuple(PredictionRecord.from_dict(item) for item in prediction_payload)
    errors = _error_records(error_payload)
    controls = tuple(ControlResult.from_dict(item) for item in control_payload)
    provenance_report = RunArtifactProvenance.from_dict(provenance)
    _validate_saved_bindings(records, run_id, provenance_report)
    _validate_error_pairing(records, errors)
    explorer = build_failure_case_explorer(
        records,
        errors,
        controls,
        prediction_artifact=prediction_artifact,
        evaluation_artifact=evaluation_artifact,
        provenance_binding_sha256=provenance_report.binding_sha256,
    )
    return _render(explorer)


def load_failure_artifact_render(
    prediction_path: str | Path,
    error_path: str | Path,
    evaluation_path: str | Path,
    provenance_path: str | Path,
) -> FailureArtifactRender:
    """Load only saved report files and render a validated failure-case review.

    The evaluation report must explicitly reference the supplied prediction
    file; its run identifier must match both prediction rows and provenance.
    """
    prediction_file = Path(prediction_path)
    evaluation_file = Path(evaluation_path)
    records = read_prediction_jsonl(prediction_file)
    errors = _error_records(_read_json(error_path, "error artifact"))
    evaluation = _evaluation_report(_read_json(evaluation_file, "evaluation report"))
    provenance = read_provenance_report(provenance_path)
    if evaluation.prediction_artifact != str(prediction_file):
        raise ProvenanceError("evaluation report prediction_artifact does not match supplied file")
    _validate_saved_bindings(records, evaluation.run_id, provenance)
    _validate_error_pairing(records, errors)
    explorer = build_failure_case_explorer(
        records,
        errors,
        evaluation.control_results,
        prediction_artifact=evaluation.prediction_artifact,
        evaluation_artifact=str(evaluation_file),
        provenance_binding_sha256=provenance.binding_sha256,
    )
    return _render(explorer)


def _render(explorer: FailureCaseExplorerArtifact) -> FailureArtifactRender:
    gallery = render_failure_case_gallery(
        [
            {
                "sample_id": case.sample_id,
                "error_category": case.error_category,
                "reference": case.reference,
                "full_prediction": case.full_prediction,
                "control_predictions": "; ".join(
                    f"{name}={value}" for name, value in sorted(case.control_predictions.items())
                )
                or "Missing",
            }
            for case in explorer.cases
        ]
    )
    return FailureArtifactRender(explorer, render_failure_case_explorer_markdown(explorer), gallery)


def _validate_saved_bindings(
    records: Sequence[PredictionRecord], run_id: str, provenance: RunArtifactProvenance
) -> None:
    if not records:
        raise ProvenanceError("saved prediction artifact must contain at least one record")
    if any(not record.target_free for record in records):
        raise ProvenanceError("failure reporting accepts only target-free saved predictions")
    record_runs = {record.run_id for record in records}
    if record_runs != {run_id} or provenance.run_id != run_id:
        raise ProvenanceError("prediction, evaluation, and provenance run IDs must match")
    if provenance.information_access.inference_target_text:
        raise ProvenanceError("provenance declares target text access at inference")


def _validate_error_pairing(
    records: Sequence[PredictionRecord], errors: Sequence[TextErrorRecord]
) -> None:
    """Ensure saved taxonomy rows describe precisely the saved full outputs."""
    full = {record.sample_id: record for record in records if record.control.value == "full"}
    if len(full) != sum(record.control.value == "full" for record in records):
        raise ValueError("full prediction sample IDs must be unique")
    error_by_id = {error.sample_id: error for error in errors}
    if len(error_by_id) != len(errors) or set(error_by_id) != set(full):
        raise ValueError("error records must uniquely match full prediction sample IDs")
    for sample_id, error in error_by_id.items():
        prediction = full[sample_id]
        if error.hypothesis != prediction.prediction_text:
            raise ValueError("error hypothesis must match saved full prediction text")
        if prediction.reference_text is not None and error.reference != prediction.reference_text:
            raise ValueError("error reference must match saved full prediction reference")


def _error_records(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[TextErrorRecord, ...]:
    rows: Any = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("error artifact must be a JSON array or an object with records")
    records: list[TextErrorRecord] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each error artifact row must be an object")
        operations = row.get("operations", {})
        if not isinstance(operations, Mapping):
            raise ValueError("error operations must be an object")
        records.append(
            TextErrorRecord(
                sample_id=str(row["sample_id"]),
                reference=str(row["reference"]),
                hypothesis=str(row["hypothesis"]),
                category=TextErrorCategory(row["category"]),
                operations=EditOperations(
                    insertions=int(operations.get("insertions", 0)),
                    deletions=int(operations.get("deletions", 0)),
                    substitutions=int(operations.get("substitutions", 0)),
                ),
            )
        )
    return tuple(records)


def _read_json(path: str | Path, artifact_name: str) -> Any:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"{artifact_name} is not valid JSON") from error


def _evaluation_report(payload: Any) -> EvaluationReport:
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation report must be a JSON object")
    return EvaluationReport.from_dict(payload)
