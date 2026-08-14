"""Load authorized precomputed feature arrays from portable JSON artifacts.

This module is intentionally separate from dataset importers: it never reads
raw participant recordings, MATLAB files, FIF files, or pickle/``torch``
payloads.  An authorized artifact maps canonical manifest sample IDs to local
JSON ``[channels, time]`` feature arrays and declares how any fitted transform
was restricted to train data.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import torch

from .json_signals import ManifestSplit, select_split_samples
from .manifest import DatasetManifest
from .prepared import TensorBackedSample
from .schema import NeuralTextSample


AUTHORIZED_FEATURE_KIND = "openthought2text.authorized_json_feature_artifact"
AUTHORIZED_FEATURE_VERSION = "1.0"
_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")


def _canonical_checksum(value: Mapping[str, Any]) -> str:
    data = dict(value)
    data.pop("checksum", None)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthorizedFeatureMapping:
    """One safe local JSON feature reference for a canonical sample."""

    sample_id: str
    split: str
    uri: str
    checksum_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuthorizedFeatureMapping":
        mapping = cls(
            sample_id=str(value["sample_id"]),
            split=str(value["split"]),
            uri=str(value["uri"]),
            checksum_sha256=str(value["checksum_sha256"]),
        )
        if not mapping.sample_id or not mapping.split:
            raise ValueError("feature mapping sample_id and split must be non-empty")
        if _CHECKSUM.fullmatch(mapping.checksum_sha256) is None:
            raise ValueError("feature mapping checksum_sha256 must be lowercase SHA-256 hex")
        return mapping

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "uri": self.uri,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthorizedFeatureArtifact:
    """Verified descriptor for precomputed JSON features and train-only fit audit."""

    authorization: str
    source_manifest_checksum: str
    mappings: tuple[AuthorizedFeatureMapping, ...]
    fit_sample_ids: tuple[str, ...]
    checksum: str
    version: str = AUTHORIZED_FEATURE_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuthorizedFeatureArtifact":
        if value.get("kind") != AUTHORIZED_FEATURE_KIND:
            raise ValueError("artifact kind is not an authorized JSON feature artifact")
        if value.get("version") != AUTHORIZED_FEATURE_VERSION:
            raise ValueError("unsupported authorized feature artifact version")
        authorization = value.get("authorization")
        source_checksum = value.get("source_manifest_checksum")
        audit = value.get("train_only_audit")
        mappings = value.get("mappings")
        checksum = value.get("checksum")
        if not isinstance(authorization, str) or not authorization.strip():
            raise ValueError("artifact authorization is required")
        if not isinstance(source_checksum, str) or _CHECKSUM.fullmatch(source_checksum) is None:
            raise ValueError("artifact source_manifest_checksum must be SHA-256 hex")
        if not isinstance(audit, Mapping) or audit.get("fit_split") != "train":
            raise ValueError("artifact train_only_audit.fit_split must be 'train'")
        fit_sample_ids = audit.get("fit_sample_ids")
        if not isinstance(fit_sample_ids, list) or not fit_sample_ids or not all(
            isinstance(item, str) and item for item in fit_sample_ids
        ):
            raise ValueError("artifact train_only_audit.fit_sample_ids must be non-empty strings")
        if len(fit_sample_ids) != len(set(fit_sample_ids)):
            raise ValueError("artifact train_only_audit.fit_sample_ids must be unique")
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("artifact mappings must be a non-empty list")
        parsed_mappings = tuple(AuthorizedFeatureMapping.from_dict(item) for item in mappings)
        if len({item.sample_id for item in parsed_mappings}) != len(parsed_mappings):
            raise ValueError("artifact mappings must have unique sample IDs")
        if not isinstance(checksum, str) or _CHECKSUM.fullmatch(checksum) is None:
            raise ValueError("artifact checksum must be lowercase SHA-256 hex")
        if checksum != _canonical_checksum(value):
            raise ValueError("artifact checksum does not match its contents")
        return cls(
            authorization=authorization,
            source_manifest_checksum=source_checksum,
            mappings=parsed_mappings,
            fit_sample_ids=tuple(fit_sample_ids),
            checksum=checksum,
        )


@dataclass(frozen=True, slots=True)
class ArtifactAuditIssue:
    code: str
    message: str
    sample_id: str | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class ArtifactAuditReport:
    artifact: AuthorizedFeatureArtifact | None
    issues: tuple[ArtifactAuditIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return self.artifact is not None and not self.issues

    def require_valid(self) -> AuthorizedFeatureArtifact:
        if not self.passed:
            codes = ", ".join(issue.code for issue in self.issues) or "invalid artifact"
            raise ValueError(f"authorized feature artifact audit failed: {codes}")
        assert self.artifact is not None
        return self.artifact


def _manifest_checksum(manifest: DatasetManifest) -> str:
    payload = {"header": manifest.header_dict(), "samples": [row.to_dict() for row in manifest.samples]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _safe_path(root: Path, uri: str) -> Path:
    candidate = Path(uri)
    if not uri or candidate.is_absolute() or ".." in candidate.parts or "://" in uri or uri.startswith("file:"):
        raise ValueError("feature URI must be a local relative path")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("feature URI escapes artifact root") from error
    if path.suffix.casefold() != ".json":
        raise ValueError("feature URI must reference a JSON array file")
    return path


def _json_matrix(path: Path, sample: NeuralTextSample) -> torch.Tensor:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"sample {sample.sample_id} feature JSON is invalid") from error
    if not isinstance(value, list) or len(value) != sample.signal.channel_count:
        raise ValueError(f"sample {sample.sample_id} feature JSON must be [expected_channels, time]")
    rows: list[list[float]] = []
    width: int | None = None
    for channel in value:
        if not isinstance(channel, list) or not channel:
            raise ValueError(f"sample {sample.sample_id} feature channel is not a non-empty list")
        width = len(channel) if width is None else width
        if len(channel) != width:
            raise ValueError(f"sample {sample.sample_id} feature JSON has uneven channel lengths")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in channel):
            raise ValueError(f"sample {sample.sample_id} feature JSON contains a non-numeric value")
        row = [float(item) for item in channel]
        if not all(torch.isfinite(torch.tensor(item)).item() for item in row):
            raise ValueError(f"sample {sample.sample_id} feature JSON contains a non-finite value")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def audit_authorized_json_features(
    manifest: DatasetManifest, descriptor: Mapping[str, Any]
) -> ArtifactAuditReport:
    """Validate mapping identity and prove fitted processing used only train samples."""
    try:
        artifact = AuthorizedFeatureArtifact.from_dict(descriptor)
    except (KeyError, TypeError, ValueError) as error:
        return ArtifactAuditReport(None, (ArtifactAuditIssue("INVALID_ARTIFACT_DESCRIPTOR", str(error)),))
    issues: list[ArtifactAuditIssue] = []
    if artifact.source_manifest_checksum != _manifest_checksum(manifest):
        issues.append(ArtifactAuditIssue("SOURCE_MANIFEST_MISMATCH", "artifact was not built for this manifest"))
    sample_by_id = {sample.sample_id: sample for sample in manifest.samples}
    for sample_id in artifact.fit_sample_ids:
        sample = sample_by_id.get(sample_id)
        if sample is None:
            issues.append(ArtifactAuditIssue("UNKNOWN_FIT_SAMPLE", "fit sample is absent from manifest", sample_id))
        elif sample.split != "train":
            issues.append(ArtifactAuditIssue("NONTRAIN_FIT_SAMPLE", "fitted processing included a non-train sample", sample_id))
    for mapping in artifact.mappings:
        sample = sample_by_id.get(mapping.sample_id)
        if sample is None:
            issues.append(ArtifactAuditIssue("UNKNOWN_MAPPED_SAMPLE", "mapping sample is absent from manifest", mapping.sample_id))
        elif sample.split != mapping.split:
            issues.append(ArtifactAuditIssue("SPLIT_MAPPING_MISMATCH", "mapping split differs from canonical manifest", mapping.sample_id))
    return ArtifactAuditReport(artifact, tuple(issues))


def load_authorized_json_features(
    manifest: DatasetManifest,
    root: str | Path,
    descriptor: Mapping[str, Any],
    *,
    split: ManifestSplit | str | None = None,
) -> tuple[TensorBackedSample, ...]:
    """Load only checksum-verified JSON arrays from a valid authorized descriptor."""
    artifact = audit_authorized_json_features(manifest, descriptor).require_valid()
    artifact_root = Path(root).expanduser().resolve()
    if not artifact_root.is_dir():
        raise ValueError("authorized feature artifact root is not a directory")
    mappings = {item.sample_id: item for item in artifact.mappings}
    samples: Iterable[NeuralTextSample] = manifest.samples
    if split is not None:
        samples = select_split_samples(samples, split)
    loaded: list[TensorBackedSample] = []
    for sample in samples:
        mapping = mappings.get(sample.sample_id)
        if mapping is None:
            raise ValueError(f"sample {sample.sample_id} has no authorized feature mapping")
        path = _safe_path(artifact_root, mapping.uri)
        if not path.is_file():
            raise ValueError(f"sample {sample.sample_id} authorized feature file is missing")
        if sha256(path.read_bytes()).hexdigest() != mapping.checksum_sha256:
            raise ValueError(f"sample {sample.sample_id} authorized feature checksum does not match")
        loaded.append(TensorBackedSample(sample=sample, values=_json_matrix(path, sample)))
    return tuple(loaded)
