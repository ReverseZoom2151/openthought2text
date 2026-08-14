"""Strict target-free evaluation execution specifications."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from openthought2text.controls import ControlCondition

from .provenance import ArtifactBinding, ProvenanceError

EXECUTION_SPEC_VERSION = "1.0"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = ("target", "label", "gold", "oracle", "reference", "stimulus_text")


@dataclass(frozen=True, slots=True)
class TargetFreeEvaluationSpec:
    preflight_plan_sha256: str
    model: ArtifactBinding
    checkpoint: ArtifactBinding
    resolved_config: ArtifactBinding
    control_conditions: tuple[ControlCondition, ...]
    # Kept structural at import time: importing ``evaluation`` initializes its
    # release-gate exports, which in turn depend on reporting provenance.
    benchmark_rows: tuple[Any, ...]
    inference_fields: tuple[str, ...]
    required_output_artifacts: tuple[str, ...]
    schema_version: str = EXECUTION_SPEC_VERSION

    def __post_init__(self) -> None:
        if not _SHA.fullmatch(self.preflight_plan_sha256):
            raise ProvenanceError("preflight_plan_sha256 must be a lowercase SHA-256 digest")
        if (
            not self.control_conditions
            or ControlCondition.FULL not in self.control_conditions
            or len(set(self.control_conditions)) != len(self.control_conditions)
        ):
            raise ProvenanceError("named controls must be unique and include full")
        if not self.benchmark_rows or len({row.value for row in self.benchmark_rows}) != len(
            self.benchmark_rows
        ):
            raise ProvenanceError("benchmark rows must be explicit and unique")
        if not self.inference_fields or any(not field.strip() for field in self.inference_fields):
            raise ProvenanceError("inference fields must be explicit")
        if any(
            any(forbidden in field.casefold() for forbidden in _FORBIDDEN)
            for field in self.inference_fields
        ):
            raise ProvenanceError(
                "inference fields cannot declare target, label, reference, or oracle access"
            )
        if not self.required_output_artifacts or any(
            not item.strip() for item in self.required_output_artifacts
        ):
            raise ProvenanceError("required output artifacts must be explicit")
        if self.schema_version != EXECUTION_SPEC_VERSION:
            raise ProvenanceError("unsupported execution spec schema version")

    @property
    def binding_sha256(self) -> str:
        return sha256(
            json.dumps(self.binding_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def binding_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preflight_plan_sha256": self.preflight_plan_sha256,
            "model": self.model.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "resolved_config": self.resolved_config.to_dict(),
            "control_conditions": [item.value for item in self.control_conditions],
            "benchmark_rows": [item.to_dict() for item in self.benchmark_rows],
            "inference_fields": list(self.inference_fields),
            "required_output_artifacts": list(self.required_output_artifacts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.binding_dict(), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetFreeEvaluationSpec:
        # Delay this import until reporting itself is fully initialized.
        from openthought2text.evaluation.records import BenchmarkRowLabel

        try:
            spec = cls(
                str(data["preflight_plan_sha256"]),
                ArtifactBinding.from_dict(data["model"]),
                ArtifactBinding.from_dict(data["checkpoint"]),
                ArtifactBinding.from_dict(data["resolved_config"]),
                tuple(ControlCondition(item) for item in data["control_conditions"]),
                tuple(BenchmarkRowLabel.from_dict(item) for item in data["benchmark_rows"]),
                tuple(data["inference_fields"]),
                tuple(data["required_output_artifacts"]),
                str(data["schema_version"]),
            )
        except ProvenanceError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ProvenanceError("invalid target-free evaluation spec schema") from error
        if data.get("binding_sha256") != spec.binding_sha256:
            raise ProvenanceError("execution spec binding_sha256 does not match contents")
        return spec
