"""Markdown-only validation for evidence bindings in generated model cards."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCardReferenceBindings:
    provenance_sha256: str
    evaluation_sha256: str
    release_gate_sha256: str


class ModelCardReferenceFailureCode(str, Enum):
    MISSING_BINDING = "missing_binding"
    MISMATCHED_BINDING = "mismatched_binding"
    CLAIMED_WITH_FAILED_GATE = "claimed_with_failed_gate"


@dataclass(frozen=True, slots=True)
class ModelCardReferenceFailure:
    code: ModelCardReferenceFailureCode
    message: str


@dataclass(frozen=True, slots=True)
class ModelCardReferenceValidation:
    failures: tuple[ModelCardReferenceFailure, ...]

    @property
    def valid(self) -> bool:
        return not self.failures


def compute_model_card_reference_bindings(
    evaluation: Any, provenance: Any, release_gate: Any
) -> ModelCardReferenceBindings:
    """Compute deterministic digests for the three evidence inputs to a model card."""
    return ModelCardReferenceBindings(
        provenance_sha256=str(provenance.binding_sha256),
        evaluation_sha256=_digest(evaluation.to_dict()),
        release_gate_sha256=_digest(_gate_payload(release_gate)),
    )


def validate_model_card_references(
    markdown: str,
    expected: ModelCardReferenceBindings,
    *,
    gate_passed: bool,
) -> ModelCardReferenceValidation:
    """Validate binding rows and claim status by parsing Markdown text only."""
    if not isinstance(markdown, str):
        raise TypeError("markdown must be text")
    found: dict[str, list[str]] = {}
    pattern = re.compile(
        r"^\|\s*(Provenance|Evaluation|Release gate) binding\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$"
    )
    for line in markdown.splitlines():
        match = pattern.fullmatch(line)
        if match:
            found.setdefault(match.group(1), []).append(match.group(2))
    expected_values = {
        "Provenance": expected.provenance_sha256,
        "Evaluation": expected.evaluation_sha256,
        "Release gate": expected.release_gate_sha256,
    }
    failures: list[ModelCardReferenceFailure] = []
    for label, digest in expected_values.items():
        values = found.get(label, [])
        if len(values) != 1:
            failures.append(
                ModelCardReferenceFailure(
                    ModelCardReferenceFailureCode.MISSING_BINDING,
                    f"Markdown must contain exactly one {label.casefold()} binding row",
                )
            )
        elif values[0] != digest:
            failures.append(
                ModelCardReferenceFailure(
                    ModelCardReferenceFailureCode.MISMATCHED_BINDING,
                    f"{label} binding digest does not match expected evidence",
                )
            )
    if not gate_passed and "**CLAIMED" in markdown:
        failures.append(
            ModelCardReferenceFailure(
                ModelCardReferenceFailureCode.CLAIMED_WITH_FAILED_GATE,
                "Markdown presents a CLAIMED status although the release gate failed",
            )
        )
    return ModelCardReferenceValidation(tuple(failures))


def _gate_payload(gate: Any) -> dict[str, Any]:
    policy = gate.policy
    return {
        "passed": gate.passed,
        "policy": {
            "primary_metric": policy.primary_metric,
            "minimum_grounded_gain": policy.minimum_grounded_gain,
            "minimum_neural_contribution": policy.minimum_neural_contribution,
            "required_controls": [str(condition.value) for condition in policy.required_controls],
            "require_target_free_information_access": policy.require_target_free_information_access,
        },
        "failures": [
            {
                "code": failure.code.value,
                "message": failure.message,
                "evidence": dict(failure.evidence),
            }
            for failure in gate.failures
        ],
    }


def _digest(value: Any) -> str:
    serialized = json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    return value
