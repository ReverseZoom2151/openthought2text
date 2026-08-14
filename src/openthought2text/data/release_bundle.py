"""Checksummed JSON release bundles binding benchmark-critical artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .authorized_features import audit_authorized_json_features
from .dataset_card import load_dataset_card
from .manifest import load_manifest
from .schema import InformationAccess
from .splits import SplitPlan, SplitProtocol, validate_split_plan

RELEASE_BUNDLE_KIND = "openthought2text.dataset_release_bundle"
RELEASE_BUNDLE_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("checksum", None)
    return sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_path(root: Path, value: str | Path) -> tuple[Path, str]:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("release artifacts must be within the release root") from error
    if not path.is_file():
        raise ValueError(f"release artifact is missing: {relative}")
    return path, relative.as_posix()


@dataclass(frozen=True, slots=True)
class ReleaseArtifactReference:
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri or Path(self.uri).is_absolute() or ".." in Path(self.uri).parts:
            raise ValueError("release artifact URI must be a local relative path")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("release artifact SHA-256 must be lowercase hex")

    def to_dict(self) -> dict[str, str]:
        return {"uri": self.uri, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReleaseArtifactReference:
        return cls(str(data["uri"]), str(data["sha256"]))


@dataclass(frozen=True, slots=True)
class DatasetReleaseBundle:
    """Cryptographic release binding; it never imports participant recordings."""

    dataset_id: str
    dataset_card: ReleaseArtifactReference
    canonical_manifest: ReleaseArtifactReference
    derived_split_plan: ReleaseArtifactReference
    authorized_feature_descriptor: ReleaseArtifactReference
    information_access: InformationAccess
    checksum: str | None = None
    version: str = RELEASE_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("release bundle dataset_id must be non-empty")
        if self.version != RELEASE_BUNDLE_VERSION:
            raise ValueError(f"unsupported release bundle version: {self.version!r}")
        if self.checksum is not None and _SHA256.fullmatch(self.checksum) is None:
            raise ValueError("release bundle checksum must be lowercase SHA-256 hex")

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": RELEASE_BUNDLE_KIND,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "dataset_card": self.dataset_card.to_dict(),
            "canonical_manifest": self.canonical_manifest.to_dict(),
            "derived_split_plan": self.derived_split_plan.to_dict(),
            "authorized_feature_descriptor": self.authorized_feature_descriptor.to_dict(),
            "information_access": self.information_access.to_dict(),
        }
        if include_checksum:
            data["checksum"] = self.checksum or _canonical_digest(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetReleaseBundle:
        if data.get("kind") != RELEASE_BUNDLE_KIND:
            raise ValueError("not an OpenThought2Text dataset release bundle")
        try:
            bundle = cls(
                dataset_id=str(data["dataset_id"]),
                dataset_card=ReleaseArtifactReference.from_dict(data["dataset_card"]),
                canonical_manifest=ReleaseArtifactReference.from_dict(data["canonical_manifest"]),
                derived_split_plan=ReleaseArtifactReference.from_dict(data["derived_split_plan"]),
                authorized_feature_descriptor=ReleaseArtifactReference.from_dict(
                    data["authorized_feature_descriptor"]
                ),
                information_access=InformationAccess.from_dict(data["information_access"]),
                checksum=str(data["checksum"]),
                version=str(data.get("version", RELEASE_BUNDLE_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid dataset release bundle schema") from error
        if bundle.checksum != _canonical_digest(data):
            raise ValueError("release bundle checksum does not match its contents")
        return bundle


@dataclass(frozen=True, slots=True)
class ReleaseBundleIssue:
    code: str
    message: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class ReleaseBundleAuditReport:
    path: Path
    bundle: DatasetReleaseBundle | None = None
    issues: tuple[ReleaseBundleIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return self.bundle is not None and not self.issues

    def require_valid(self) -> DatasetReleaseBundle:
        if not self.passed:
            raise ValueError(
                "release bundle audit failed: " + ", ".join(x.code for x in self.issues)
            )
        assert self.bundle is not None
        return self.bundle


def _load_plan(path: Path) -> SplitPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("assignments"), list):
        raise ValueError("split plan must be an object with assignments")
    return SplitPlan(
        protocol=SplitProtocol(data["protocol"]),
        seed=int(data["seed"]),
        assignments=tuple(
            (str(row["sample_id"]), str(row["split"])) for row in data["assignments"]
        ),
        excluded_sample_ids=tuple(str(item) for item in data.get("excluded_sample_ids", [])),
        held_out_subject=data.get("held_out_subject"),
    )


def _reference(root: Path, value: str | Path) -> ReleaseArtifactReference:
    path, uri = _safe_path(root, value)
    return ReleaseArtifactReference(uri, _digest(path))


def build_dataset_release_bundle(
    root: str | Path,
    *,
    dataset_card: str | Path,
    canonical_manifest: str | Path,
    derived_split_plan: str | Path,
    authorized_feature_descriptor: str | Path,
) -> DatasetReleaseBundle:
    """Validate bound JSON artifacts, then create the immutable release binding."""
    release_root = Path(root).expanduser().resolve()
    if not release_root.is_dir():
        raise ValueError("release root is not a directory")
    card_path, _ = _safe_path(release_root, dataset_card)
    manifest_path, _ = _safe_path(release_root, canonical_manifest)
    plan_path, _ = _safe_path(release_root, derived_split_plan)
    features_path, _ = _safe_path(release_root, authorized_feature_descriptor)
    card = load_dataset_card(card_path)
    manifest = load_manifest(manifest_path)
    if card.dataset_id != manifest.dataset_id:
        raise ValueError("dataset card and canonical manifest dataset_id differ")
    plan = _load_plan(plan_path)
    validate_split_plan(manifest.samples, plan).require_valid()
    try:
        descriptor = json.loads(features_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("authorized feature descriptor must be valid JSON") from error
    if not isinstance(descriptor, Mapping):
        raise ValueError("authorized feature descriptor must be a JSON object")
    audit_authorized_json_features(manifest, descriptor).require_valid()
    return DatasetReleaseBundle(
        dataset_id=manifest.dataset_id,
        dataset_card=_reference(release_root, card_path),
        canonical_manifest=_reference(release_root, manifest_path),
        derived_split_plan=_reference(release_root, plan_path),
        authorized_feature_descriptor=_reference(release_root, features_path),
        information_access=manifest.information_access,
    )


def write_dataset_release_bundle(path: str | Path, bundle: DatasetReleaseBundle) -> None:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("release bundles must be written as .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(bundle.to_dict(), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def audit_dataset_release_bundle(path: str | Path) -> ReleaseBundleAuditReport:
    """Revalidate a bundle, checksums, source policy, and every bound artifact."""
    bundle_path = Path(path).expanduser().resolve()
    if bundle_path.suffix.casefold() != ".json":
        return ReleaseBundleAuditReport(
            bundle_path, issues=(ReleaseBundleIssue("UNSUPPORTED_FORMAT", "bundle must use .json"),)
        )
    try:
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("bundle must be a JSON object")
        bundle = DatasetReleaseBundle.from_dict(raw)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return ReleaseBundleAuditReport(
            bundle_path, issues=(ReleaseBundleIssue("INVALID_BUNDLE", str(error), bundle_path),)
        )
    root = bundle_path.parent
    references = {
        "dataset_card": bundle.dataset_card,
        "canonical_manifest": bundle.canonical_manifest,
        "derived_split_plan": bundle.derived_split_plan,
        "authorized_feature_descriptor": bundle.authorized_feature_descriptor,
    }
    issues: list[ReleaseBundleIssue] = []
    paths: dict[str, Path] = {}
    for name, reference in references.items():
        try:
            artifact_path, _ = _safe_path(root, reference.uri)
        except ValueError as error:
            issues.append(ReleaseBundleIssue("MISSING_OR_NONLOCAL_ARTIFACT", f"{name}: {error}"))
            continue
        paths[name] = artifact_path
        if _digest(artifact_path) != reference.sha256:
            issues.append(
                ReleaseBundleIssue(
                    "ARTIFACT_CHECKSUM_MISMATCH", f"{name} bytes changed", artifact_path
                )
            )
    if not issues:
        try:
            rebuilt = build_dataset_release_bundle(root, **paths)
            if rebuilt.information_access != bundle.information_access:
                issues.append(
                    ReleaseBundleIssue(
                        "INFORMATION_ACCESS_MISMATCH", "bundle contract differs from manifest"
                    )
                )
            if rebuilt.dataset_id != bundle.dataset_id:
                issues.append(
                    ReleaseBundleIssue(
                        "DATASET_ID_MISMATCH", "bundle dataset_id differs from manifest"
                    )
                )
        except ValueError as error:
            issues.append(ReleaseBundleIssue("INVALID_BOUND_ARTIFACT", str(error)))
    return ReleaseBundleAuditReport(bundle_path, bundle, tuple(issues))
