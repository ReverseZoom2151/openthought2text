"""Metadata-only validation for imported foundation checkpoint disclosures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .foundation_wrapper import FoundationFeatureContract, FoundationPretrainingProvenance


@dataclass(frozen=True)
class FoundationManifestCompatibility:
    compatible: bool
    errors: tuple[str, ...]
    observed_hash: str | None


def validate_foundation_checkpoint_manifest(
    manifest: Mapping[str, Any],
    file_byte_hash: str,
    contract: FoundationFeatureContract,
    provenance: FoundationPretrainingProvenance,
    frozen_intent: bool,
    allowed_key_schema_summary: tuple[str, ...],
) -> FoundationManifestCompatibility:
    """Compare plain caller metadata only; never opens/deserializes checkpoint bytes."""
    errors = []
    observed = None
    if not isinstance(manifest, Mapping):
        return FoundationManifestCompatibility(False, ("manifest must be a mapping",), None)
    if (
        not isinstance(file_byte_hash, str)
        or len(file_byte_hash) != 64
        or any(x not in "0123456789abcdef" for x in file_byte_hash.lower())
    ):
        errors.append("file_byte_hash must be a SHA-256 hex string")
    observed = manifest.get("file_byte_hash")
    if not isinstance(observed, str):
        errors.append("manifest file_byte_hash disclosure is required")
    elif observed != file_byte_hash:
        errors.append("manifest file_byte_hash differs from caller-supplied hash")
    for key, expected in (
        ("source_name", provenance.source_name),
        ("overlap_label", provenance.overlap_label),
        ("input_feature_size", contract.input_feature_size),
        ("output_feature_size", contract.output_feature_size),
        ("frozen_intent", frozen_intent),
    ):
        if key not in manifest:
            errors.append(f"manifest {key} disclosure is required")
        elif manifest[key] != expected:
            errors.append(f"manifest {key} differs from declared contract/provenance")
    if not isinstance(manifest.get("license"), str) or not manifest["license"].strip():
        errors.append("manifest license disclosure is required")
    keys = manifest.get("key_schema_summary")
    if not isinstance(keys, (list, tuple)) or any(not isinstance(x, str) for x in keys):
        errors.append("manifest key_schema_summary disclosure is required")
    elif tuple(keys) != allowed_key_schema_summary:
        errors.append("manifest key_schema_summary is not allowed")
    return FoundationManifestCompatibility(
        not errors, tuple(errors), observed if isinstance(observed, str) else None
    )
