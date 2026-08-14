"""Build canonical manifests from authorized reader mappings, never raw files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .manifest import DatasetManifest
from .schema import (
    InformationAccess,
    Modality,
    NeuralTextSample,
    SignalReference,
    TextTarget,
    TimeInterval,
)

_RAW_EXTENSIONS = {".mat", ".h5", ".hdf5", ".fif", ".edf"}


def _manifest_checksum(manifest: DatasetManifest) -> str:
    payload = {
        "header": manifest.header_dict(),
        "samples": [row.to_dict() for row in manifest.samples],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _safe_signal(value: object) -> SignalReference:
    signal = value if isinstance(value, SignalReference) else SignalReference.from_dict(value)  # type: ignore[arg-type]
    path = Path(signal.uri)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "://" in signal.uri
        or path.suffix.casefold() in _RAW_EXTENSIONS
    ):
        raise ValueError("canonical conversion requires a safe non-raw relative SignalReference")
    return signal


def _target(value: object) -> TextTarget | None:
    if value is None:
        return None
    return value if isinstance(value, TextTarget) else TextTarget.from_dict(value)  # type: ignore[arg-type]


def _target_allowed(split: str, access: InformationAccess) -> bool:
    return (
        access.train_target_text
        if split == "train"
        else access.validation_target_text
        if split == "validation"
        else False
    )


@dataclass(frozen=True, slots=True)
class CanonicalConversionArtifact:
    """Manifest plus a checksum binding it to metadata-only source provenance."""

    manifest: DatasetManifest
    manifest_checksum: str
    source_plan_checksum: str
    source_kind: str
    provenance_mapping: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.manifest_checksum != _manifest_checksum(self.manifest):
            raise ValueError("canonical conversion artifact manifest checksum does not match")
        if len(self.provenance_mapping) != len(self.manifest.samples):
            raise ValueError("conversion provenance must map every canonical sample")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "openthought2text.canonical_conversion_artifact",
            "dataset_id": self.manifest.dataset_id,
            "manifest_checksum": self.manifest_checksum,
            "source_plan_checksum": self.source_plan_checksum,
            "source_kind": self.source_kind,
            "provenance_mapping": dict(self.provenance_mapping),
        }


def build_canonical_conversion_artifact(
    dataset_id: str,
    records: Iterable[Mapping[str, Any]],
    *,
    information_access: InformationAccess,
    source_plan: Mapping[str, Any],
) -> CanonicalConversionArtifact:
    """Convert plain reader metadata to samples; it never opens signal URIs."""
    if not dataset_id.strip() or information_access.inference_target_text:
        raise ValueError("canonical conversion needs dataset_id and target-free inference access")
    if not isinstance(source_plan.get("checksum"), str) or not isinstance(
        source_plan.get("kind"), str
    ):
        raise ValueError("source plan must provide kind and checksum provenance")
    samples: list[NeuralTextSample] = []
    mapping: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"canonical record {index} must be a mapping")
        split = record.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"canonical record {index} needs train, validation, or test split")
        target = _target(record.get("target"))
        allowed = _target_allowed(split, information_access)
        if allowed and target is None:
            raise ValueError(
                f"canonical {split} record {index} requires target under declared access"
            )
        if not allowed and target is not None:
            raise ValueError(f"canonical {split}/inference record {index} must omit target text")
        signal = _safe_signal(record.get("signal"))
        try:
            interval = (
                record["interval"]
                if isinstance(record["interval"], TimeInterval)
                else TimeInterval.from_dict(record["interval"])
            )
            sample = NeuralTextSample(
                sample_id=str(record["sample_id"]),
                dataset_id=dataset_id,
                subject_id=str(record["subject_id"]),
                signal=signal,
                interval=interval,
                modality=Modality(record["modality"]),
                target=target,
                split=split,
                session_id=record.get("session_id"),
                run_id=record.get("run_id"),
                trial_id=record.get("trial_id"),
                group_ids=tuple(record.get("group_ids", ())),
                task=str(record.get("task", "unknown")),
                metadata=dict(record.get("metadata", {})),
            )
            source_record_id = record["source_record_id"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid canonical conversion record {index}") from error
        if not isinstance(source_record_id, str) or not source_record_id.strip():
            raise ValueError(f"canonical record {index} needs source_record_id provenance")
        if sample.sample_id in mapping:
            raise ValueError("canonical conversion sample_id values must be unique")
        samples.append(sample)
        mapping[sample.sample_id] = source_record_id
    if not samples:
        raise ValueError("canonical conversion requires at least one record")
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        samples=tuple(samples),
        information_access=information_access,
        description="Authorized metadata-first conversion; no raw participant payload persisted.",
        metadata={
            "source_plan_checksum": source_plan["checksum"],
            "source_kind": source_plan["kind"],
            "target_policy": "train_validation_only",
        },
    )
    return CanonicalConversionArtifact(
        manifest,
        _manifest_checksum(manifest),
        str(source_plan["checksum"]),
        str(source_plan["kind"]),
        mapping,
    )


def build_zuco_canonical_artifact(
    records: Iterable[Mapping[str, Any]],
    *,
    information_access: InformationAccess,
    source_plan: Mapping[str, Any],
) -> CanonicalConversionArtifact:
    return build_canonical_conversion_artifact(
        "zuco", records, information_access=information_access, source_plan=source_plan
    )


def build_brain2qwerty_canonical_artifact(
    records: Iterable[Mapping[str, Any]],
    *,
    information_access: InformationAccess,
    source_plan: Mapping[str, Any],
) -> CanonicalConversionArtifact:
    return build_canonical_conversion_artifact(
        "brain2qwerty", records, information_access=information_access, source_plan=source_plan
    )


def build_t15_canonical_artifact(
    records: Iterable[Mapping[str, Any]],
    *,
    information_access: InformationAccess,
    source_plan: Mapping[str, Any],
) -> CanonicalConversionArtifact:
    return build_canonical_conversion_artifact(
        "t15", records, information_access=information_access, source_plan=source_plan
    )
