"""JSON-only authorization preflight plans; they never open signal payloads."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .dataset_card import load_dataset_card
from .release_bundle import ReleaseArtifactReference, audit_dataset_release_bundle
from .schema import InformationAccess
from .splits import SplitProtocol

PREFLIGHT_PLAN_KIND = "openthought2text.authorized_dataset_preflight"
PREFLIGHT_PLAN_VERSION = "1.0"
_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("checksum", None)
    return sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_artifact(root: Path, value: str | Path) -> ReleaseArtifactReference:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        uri = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("preflight-bound artifacts must be within the plan root") from error
    if not resolved.is_file():
        raise ValueError(f"preflight-bound artifact is missing: {uri}")
    return ReleaseArtifactReference(uri, _file_digest(resolved))


def _source_root_identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_root_identifier must be a non-empty authorization identifier")
    if any(token in value for token in ("/", "\\", "..", "file:")):
        raise ValueError("source_root_identifier must not be a filesystem path")
    return value


@dataclass(frozen=True, slots=True)
class AuthorizedDatasetPreflightPlan:
    dataset_id: str
    authorization_id: str
    source_root_identifier: str
    dataset_card: ReleaseArtifactReference
    release_bundle: ReleaseArtifactReference
    split_plan: ReleaseArtifactReference
    inference_access: InformationAccess
    requested_protocols: tuple[SplitProtocol, ...]
    checksum: str | None = None
    version: str = PREFLIGHT_PLAN_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.authorization_id.strip():
            raise ValueError("preflight plan requires dataset_id and authorization_id")
        _source_root_identifier(self.source_root_identifier)
        if self.version != PREFLIGHT_PLAN_VERSION:
            raise ValueError(f"unsupported preflight plan version: {self.version!r}")
        if self.inference_access.inference_label_leakage:
            raise ValueError("preflight inference access cannot expose text labels or context")
        if not self.requested_protocols or len(set(self.requested_protocols)) != len(
            self.requested_protocols
        ):
            raise ValueError("requested_protocols must be a non-empty unique list")
        if self.checksum is not None and _CHECKSUM.fullmatch(self.checksum) is None:
            raise ValueError("preflight plan checksum must be lowercase SHA-256 hex")

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": PREFLIGHT_PLAN_KIND,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "authorization_id": self.authorization_id,
            "source_root_identifier": self.source_root_identifier,
            "dataset_card": self.dataset_card.to_dict(),
            "release_bundle": self.release_bundle.to_dict(),
            "split_plan": self.split_plan.to_dict(),
            "inference_access": self.inference_access.to_dict(),
            "requested_protocols": [protocol.value for protocol in self.requested_protocols],
        }
        if include_checksum:
            data["checksum"] = self.checksum or _canonical_digest(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuthorizedDatasetPreflightPlan:
        if data.get("kind") != PREFLIGHT_PLAN_KIND:
            raise ValueError("not an authorized dataset preflight plan")
        try:
            plan = cls(
                dataset_id=str(data["dataset_id"]),
                authorization_id=str(data["authorization_id"]),
                source_root_identifier=_source_root_identifier(str(data["source_root_identifier"])),
                dataset_card=ReleaseArtifactReference.from_dict(data["dataset_card"]),
                release_bundle=ReleaseArtifactReference.from_dict(data["release_bundle"]),
                split_plan=ReleaseArtifactReference.from_dict(data["split_plan"]),
                inference_access=InformationAccess.from_dict(data["inference_access"]),
                requested_protocols=tuple(
                    SplitProtocol(item) for item in data["requested_protocols"]
                ),
                checksum=str(data["checksum"]),
                version=str(data.get("version", PREFLIGHT_PLAN_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid preflight plan schema") from error
        if plan.checksum != _canonical_digest(data):
            raise ValueError("preflight plan checksum does not match its contents")
        return plan


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: str
    message: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    path: Path
    plan: AuthorizedDatasetPreflightPlan | None = None
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return self.plan is not None and not self.issues

    def require_valid(self) -> AuthorizedDatasetPreflightPlan:
        if not self.passed:
            raise ValueError("preflight failed: " + ", ".join(item.code for item in self.issues))
        assert self.plan is not None
        return self.plan


def build_authorized_preflight_plan(
    root: str | Path,
    *,
    dataset_card: str | Path,
    release_bundle: str | Path,
    split_plan: str | Path,
    authorization_id: str,
    source_root_identifier: str,
    requested_protocols: tuple[SplitProtocol | str, ...],
) -> AuthorizedDatasetPreflightPlan:
    """Validate metadata artifacts only; raw source data is intentionally out of scope."""
    plan_root = Path(root).expanduser().resolve()
    if not plan_root.is_dir():
        raise ValueError("preflight root is not a directory")
    card_reference = _safe_artifact(plan_root, dataset_card)
    bundle_reference = _safe_artifact(plan_root, release_bundle)
    split_reference = _safe_artifact(plan_root, split_plan)
    card = load_dataset_card(plan_root / card_reference.uri)
    bundle_report = audit_dataset_release_bundle(plan_root / bundle_reference.uri)
    bundle = bundle_report.require_valid()
    if card.dataset_id != bundle.dataset_id:
        raise ValueError("dataset card and release bundle dataset_id differ")
    if split_reference != bundle.derived_split_plan:
        raise ValueError("preflight split plan must match the release bundle binding")
    protocols = tuple(SplitProtocol(value) for value in requested_protocols)
    return AuthorizedDatasetPreflightPlan(
        dataset_id=bundle.dataset_id,
        authorization_id=authorization_id,
        source_root_identifier=_source_root_identifier(source_root_identifier),
        dataset_card=card_reference,
        release_bundle=bundle_reference,
        split_plan=split_reference,
        inference_access=bundle.information_access,
        requested_protocols=protocols,
    )


def write_authorized_preflight_plan(path: str | Path, plan: AuthorizedDatasetPreflightPlan) -> None:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("preflight plans must be written as .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def audit_authorized_preflight_plan(path: str | Path) -> PreflightReport:
    """Audit file bindings and disclosure readiness without loading signals."""
    plan_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("preflight plan must be a JSON object")
        plan = AuthorizedDatasetPreflightPlan.from_dict(raw)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return PreflightReport(
            plan_path, issues=(PreflightIssue("INVALID_PREFLIGHT_PLAN", str(error), plan_path),)
        )
    root = plan_path.parent
    refs = {
        "dataset_card": plan.dataset_card,
        "release_bundle": plan.release_bundle,
        "split_plan": plan.split_plan,
    }
    issues: list[PreflightIssue] = []
    paths: dict[str, Path] = {}
    for name, reference in refs.items():
        try:
            resolved = (root / reference.uri).resolve()
            resolved.relative_to(root)
            if not resolved.is_file():
                raise ValueError("missing artifact")
        except ValueError:
            issues.append(PreflightIssue("MISSING_OR_NONLOCAL_ARTIFACT", name))
            continue
        paths[name] = resolved
        if _file_digest(resolved) != reference.sha256:
            issues.append(PreflightIssue("ARTIFACT_CHECKSUM_MISMATCH", name, resolved))
    if not issues:
        try:
            rebuilt = build_authorized_preflight_plan(
                root,
                dataset_card=paths["dataset_card"],
                release_bundle=paths["release_bundle"],
                split_plan=paths["split_plan"],
                authorization_id=plan.authorization_id,
                source_root_identifier=plan.source_root_identifier,
                requested_protocols=tuple(plan.requested_protocols),
            )
            if rebuilt.inference_access != plan.inference_access:
                issues.append(
                    PreflightIssue("INFERENCE_ACCESS_MISMATCH", "plan differs from release bundle")
                )
        except ValueError as error:
            issues.append(PreflightIssue("NOT_RELEASE_READY", str(error)))
    return PreflightReport(plan_path, plan, tuple(issues))
