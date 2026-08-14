"""Versioned, hash-bound provenance reports for evaluable research runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

PROVENANCE_REPORT_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AMBIGUOUS = frozenset(
    {"", "-", "n/a", "na", "none", "null", "tbd", "unknown", "unset", "ambiguous"}
)


class ProvenanceError(ValueError):
    """A report does not bind a run to unambiguous, auditable inputs."""


def _required(value: str, field_name: str) -> str:
    if not isinstance(value, str) or value.strip().casefold() in _AMBIGUOUS:
        raise ProvenanceError(f"{field_name} must be explicit and non-ambiguous")
    return value


def _checksum(value: str, field_name: str) -> str:
    value = _required(value, field_name).casefold()
    if not _SHA256_RE.fullmatch(value):
        raise ProvenanceError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """Stable identity, location, and content hash of an artifact used by a run."""

    identifier: str
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        _required(self.identifier, "artifact.identifier")
        _required(self.uri, "artifact.uri")
        object.__setattr__(self, "sha256", _checksum(self.sha256, "artifact.sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"identifier": self.identifier, "uri": self.uri, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactBinding:
        _require_keys(data, {"identifier", "uri", "sha256"}, "artifact")
        return cls(
            identifier=str(data["identifier"]), uri=str(data["uri"]), sha256=str(data["sha256"])
        )


@dataclass(frozen=True, slots=True)
class InformationAccessContract:
    """All fields visible at each stage, with no implicit inference assumptions."""

    train_target_text: bool
    validation_target_text: bool
    inference_target_text: bool
    inference_text_context: bool
    inference_token_boundaries: bool
    inference_event_boundaries: bool
    inference_stimulus_audio: bool
    split_definition: str
    alignment_source: str

    def __post_init__(self) -> None:
        for name in (
            "train_target_text",
            "validation_target_text",
            "inference_target_text",
            "inference_text_context",
            "inference_token_boundaries",
            "inference_event_boundaries",
            "inference_stimulus_audio",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProvenanceError(f"information_access.{name} must be an explicit boolean")
        _required(self.split_definition, "information_access.split_definition")
        _required(self.alignment_source, "information_access.alignment_source")

    def to_dict(self) -> dict[str, bool | str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InformationAccessContract:
        fields = set(cls.__dataclass_fields__)
        _require_keys(data, fields, "information_access")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class RunArtifactProvenance:
    """A tamper-evident provenance report associated with one evaluation run."""

    run_id: str
    model: ArtifactBinding
    checkpoint: ArtifactBinding
    data_manifest: ArtifactBinding
    split_plan: ArtifactBinding
    config: ArtifactBinding
    code_revision: str
    information_access: InformationAccessContract
    schema_version: str = PROVENANCE_REPORT_VERSION

    def __post_init__(self) -> None:
        _required(self.run_id, "run_id")
        _required(self.code_revision, "code_revision")
        if self.schema_version != PROVENANCE_REPORT_VERSION:
            raise ProvenanceError(f"unsupported provenance report version: {self.schema_version!r}")

    @property
    def binding_sha256(self) -> str:
        """Digest of every research-relevant input, not of an arbitrary timestamp."""
        payload = json.dumps(self.binding_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(payload).hexdigest()

    def binding_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": self.model.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "data_manifest": self.data_manifest.to_dict(),
            "split_plan": self.split_plan.to_dict(),
            "config": self.config.to_dict(),
            "code_revision": self.code_revision,
            "information_access": self.information_access.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.binding_dict(), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunArtifactProvenance:
        required = {
            "schema_version",
            "run_id",
            "model",
            "checkpoint",
            "data_manifest",
            "split_plan",
            "config",
            "code_revision",
            "information_access",
            "binding_sha256",
        }
        _require_keys(data, required, "provenance report")
        report = cls(
            run_id=str(data["run_id"]),
            model=ArtifactBinding.from_dict(data["model"]),
            checkpoint=ArtifactBinding.from_dict(data["checkpoint"]),
            data_manifest=ArtifactBinding.from_dict(data["data_manifest"]),
            split_plan=ArtifactBinding.from_dict(data["split_plan"]),
            config=ArtifactBinding.from_dict(data["config"]),
            code_revision=str(data["code_revision"]),
            information_access=InformationAccessContract.from_dict(data["information_access"]),
            schema_version=str(data["schema_version"]),
        )
        supplied = _checksum(str(data["binding_sha256"]), "binding_sha256")
        if supplied != report.binding_sha256:
            raise ProvenanceError("binding_sha256 does not match the report's artifact bindings")
        return report


def write_provenance_report(path: str | Path, report: RunArtifactProvenance) -> None:
    """Write the portable JSON provenance report for an evaluation artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def read_provenance_report(path: str | Path) -> RunArtifactProvenance:
    with Path(path).open(encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as error:
            raise ProvenanceError("provenance report is not valid JSON") from error
    try:
        return RunArtifactProvenance.from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ProvenanceError):
            raise
        raise ProvenanceError("provenance report violates the required contract") from error


def _require_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected.difference(data)
    unexpected = set(data).difference(expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        raise ProvenanceError(f"{name} fields are not explicit ({'; '.join(details)})")
