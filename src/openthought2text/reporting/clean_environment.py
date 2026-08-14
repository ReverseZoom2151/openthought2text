"""Clean-environment reproducibility audit records; no execution is performed here."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CleanEnvironmentAuditPlan:
    python_requirement: str
    package_requirement: str
    resolved_config_sha256: str
    test_command: str
    expected_artifact_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if any(
            not isinstance(getattr(self, field), str) or not getattr(self, field).strip()
            for field in ("python_requirement", "package_requirement", "test_command")
        ) or not _SHA.fullmatch(self.resolved_config_sha256):
            raise ValueError(
                "audit plan requires explicit Python/package/command and config checksum"
            )
        if not self.expected_artifact_sha256 or any(
            not name.strip() or not _SHA.fullmatch(value)
            for name, value in self.expected_artifact_sha256.items()
        ):
            raise ValueError("expected artifacts require named lowercase SHA-256 checksums")

    @property
    def binding_sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_requirement": self.python_requirement,
            "package_requirement": self.package_requirement,
            "resolved_config_sha256": self.resolved_config_sha256,
            "test_command": self.test_command,
            "expected_artifact_sha256": dict(sorted(self.expected_artifact_sha256.items())),
        }


@dataclass(frozen=True, slots=True)
class CleanEnvironmentAuditRecord:
    plan_binding_sha256: str
    actual_environment: Mapping[str, str]
    observed_artifact_sha256: Mapping[str, str]
    matching_artifacts: tuple[str, ...]
    mismatched_artifacts: tuple[str, ...]
    no_real_data_claim: str = "Environment audit only; no successful real-data run is claimed."

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_binding_sha256": self.plan_binding_sha256,
            "actual_environment": dict(sorted(self.actual_environment.items())),
            "observed_artifact_sha256": dict(sorted(self.observed_artifact_sha256.items())),
            "matching_artifacts": list(self.matching_artifacts),
            "mismatched_artifacts": list(self.mismatched_artifacts),
            "no_real_data_claim": self.no_real_data_claim,
            "binding_sha256": _digest(self._without_binding()),
        }

    def _without_binding(self):
        return {
            "plan_binding_sha256": self.plan_binding_sha256,
            "actual_environment": dict(sorted(self.actual_environment.items())),
            "observed_artifact_sha256": dict(sorted(self.observed_artifact_sha256.items())),
            "matching_artifacts": list(self.matching_artifacts),
            "mismatched_artifacts": list(self.mismatched_artifacts),
            "no_real_data_claim": self.no_real_data_claim,
        }


def audit_clean_environment(
    plan: CleanEnvironmentAuditPlan,
    actual_environment: Mapping[str, str],
    observed_artifact_sha256: Mapping[str, str],
) -> CleanEnvironmentAuditRecord:
    """Compare caller-supplied observations to an immutable audit plan; never runs commands."""
    if not actual_environment or any(
        not key.strip() or not str(value).strip() for key, value in actual_environment.items()
    ):
        raise ValueError("actual environment inputs must be explicit")
    matching = tuple(
        sorted(
            name
            for name, expected in plan.expected_artifact_sha256.items()
            if observed_artifact_sha256.get(name) == expected
        )
    )
    mismatched = tuple(
        sorted(name for name in plan.expected_artifact_sha256 if name not in matching)
    )
    return CleanEnvironmentAuditRecord(
        plan.binding_sha256,
        dict(actual_environment),
        dict(observed_artifact_sha256),
        matching,
        mismatched,
    )


def render_clean_environment_markdown(record: CleanEnvironmentAuditRecord) -> str:
    return f"# Clean-environment audit\n\n**{record.no_real_data_claim}**\n\nPlan binding: `{record.plan_binding_sha256}`\n\n- Matching artifacts: {', '.join(record.matching_artifacts) or 'none'}\n- Mismatched artifacts: {', '.join(record.mismatched_artifacts) or 'none'}\n"


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
