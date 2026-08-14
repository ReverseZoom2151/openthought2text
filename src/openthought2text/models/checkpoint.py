"""Metadata-only architecture compatibility checks for model checkpoints.

This module intentionally never calls ``torch.load`` or opens a path.  Callers
must supply already-decoded, trusted metadata; this helper only compares plain
architecture data against the model they have constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from .factory import architecture_fingerprint, describe_model_architecture
from .model import NeuralToTextModel


@dataclass(frozen=True)
class CheckpointArchitectureCompatibility:
    """Structured outcome for a checkpoint/model architecture comparison."""

    compatible: bool
    errors: tuple[str, ...]
    expected_fingerprint: str
    observed_fingerprint: str | None
    expected_description: dict[str, Any]
    observed_description: dict[str, Any] | None

    def raise_if_incompatible(self) -> None:
        if not self.compatible:
            detail = "; ".join(self.errors) or "unknown incompatibility"
            raise ValueError(f"checkpoint architecture is incompatible: {detail}")


def checkpoint_architecture_metadata(model: NeuralToTextModel) -> dict[str, Any]:
    """Build serializable architecture metadata to save alongside a checkpoint."""
    description = describe_model_architecture(model)
    return {
        "architecture_format_version": 1,
        "architecture_fingerprint": architecture_fingerprint(model),
        "architecture_description": description,
    }


def _canonical_fingerprint(description: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(description), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _field_mismatches(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> list[str]:
    """Compact, useful differences without recursively dumping a whole schema."""
    errors: list[str] = []
    if expected.get("format_version") != observed.get("format_version"):
        errors.append(
            f"description format_version differs (checkpoint={observed.get('format_version')!r}, "
            f"model={expected.get('format_version')!r})"
        )
    observed_config = observed.get("config")
    expected_config = expected.get("config")
    if not isinstance(observed_config, Mapping):
        errors.append("checkpoint architecture_description.config must be a mapping")
    elif isinstance(expected_config, Mapping):
        for key in sorted(set(expected_config) | set(observed_config)):
            if expected_config.get(key) != observed_config.get(key):
                errors.append(
                    f"config.{key} differs (checkpoint={observed_config.get(key)!r}, "
                    f"model={expected_config.get(key)!r})"
                )
    if expected.get("state_dict_schema") != observed.get("state_dict_schema"):
        errors.append("state_dict schema differs")
    return errors


def validate_checkpoint_architecture(
    model: NeuralToTextModel,
    saved_metadata: Mapping[str, Any],
) -> CheckpointArchitectureCompatibility:
    """Compare plain saved metadata with ``model`` before loading state tensors.

    The required metadata keys are ``architecture_fingerprint`` and
    ``architecture_description``.  A malformed description is reported as an
    incompatibility rather than being deserialized or executed.
    """
    expected_description = describe_model_architecture(model)
    expected_fingerprint = architecture_fingerprint(model)
    errors: list[str] = []
    observed_fingerprint: str | None = None
    observed_description: dict[str, Any] | None = None
    if not isinstance(saved_metadata, Mapping):
        errors.append("saved_metadata must be a mapping")
    else:
        raw_fingerprint = saved_metadata.get("architecture_fingerprint")
        if not isinstance(raw_fingerprint, str):
            errors.append("checkpoint architecture_fingerprint must be a string")
        else:
            observed_fingerprint = raw_fingerprint
            if observed_fingerprint != expected_fingerprint:
                errors.append("architecture fingerprint differs")
        raw_description = saved_metadata.get("architecture_description")
        if not isinstance(raw_description, Mapping):
            errors.append("checkpoint architecture_description must be a mapping")
        else:
            observed_description = dict(raw_description)
            try:
                metadata_fingerprint = _canonical_fingerprint(observed_description)
            except (TypeError, ValueError) as error:
                errors.append(f"checkpoint architecture_description is not canonical JSON: {error}")
            else:
                if observed_fingerprint is not None and metadata_fingerprint != observed_fingerprint:
                    errors.append("checkpoint fingerprint does not match its architecture_description")
            errors.extend(_field_mismatches(expected_description, observed_description))
    return CheckpointArchitectureCompatibility(
        compatible=not errors,
        errors=tuple(errors),
        expected_fingerprint=expected_fingerprint,
        observed_fingerprint=observed_fingerprint,
        expected_description=expected_description,
        observed_description=observed_description,
    )
