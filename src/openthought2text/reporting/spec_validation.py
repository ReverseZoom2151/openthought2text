"""File-backed, target-free execution-spec validation for operational CLIs.

These helpers intentionally inspect only the serialized plan.  They never
resolve model, checkpoint, dataset, or prediction paths, so a CLI can expose
preflight validation without accidentally starting an evaluation run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openthought2text.controls import ControlCondition

from .execution_spec import TargetFreeEvaluationSpec
from .operational_evaluation import (
    ControlSuitePlanValidation,
    render_control_suite_plan_markdown,
    validate_complete_control_suite_plan,
)


@dataclass(frozen=True, slots=True)
class SerializedSpecValidation:
    """A machine-readable and Markdown-ready preflight result."""

    source_path: str | None
    execution_spec_binding_sha256: str
    control_suite: ControlSuitePlanValidation

    @property
    def valid(self) -> bool:
        return self.control_suite.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "execution_spec_binding_sha256": self.execution_spec_binding_sha256,
            "valid": self.valid,
            "control_suite": self.control_suite.to_dict(),
            "no_performance_claim": "Plan validation only; no evaluation was executed.",
        }

    def to_markdown(self) -> str:
        return render_control_suite_plan_markdown(self.control_suite)


def validate_serialized_target_free_spec(
    payload: Mapping[str, Any],
    *,
    required_controls: Sequence[ControlCondition] = tuple(ControlCondition),
    required_outputs: Sequence[str] = (
        "predictions.jsonl",
        "evaluation.json",
        "provenance.json",
    ),
) -> SerializedSpecValidation:
    """Purely validate a decoded execution-spec mapping and its control plan."""
    spec = TargetFreeEvaluationSpec.from_dict(payload)
    control_suite = validate_complete_control_suite_plan(
        spec,
        required_controls=required_controls,
        required_outputs=required_outputs,
    )
    return SerializedSpecValidation(None, spec.binding_sha256, control_suite)


def validate_target_free_spec_file(
    path: str | Path,
    *,
    required_controls: Sequence[ControlCondition] = tuple(ControlCondition),
    required_outputs: Sequence[str] = (
        "predictions.jsonl",
        "evaluation.json",
        "provenance.json",
    ),
) -> SerializedSpecValidation:
    """Load a JSON execution spec and validate its declared complete control suite."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("execution spec JSON must contain an object")
    result = validate_serialized_target_free_spec(
        payload,
        required_controls=required_controls,
        required_outputs=required_outputs,
    )
    return SerializedSpecValidation(
        str(source), result.execution_spec_binding_sha256, result.control_suite
    )
