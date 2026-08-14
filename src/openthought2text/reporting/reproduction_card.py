"""Machine-validated provenance cards for corrected baseline reproductions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .provenance import ArtifactBinding, ProvenanceError

REPRODUCTION_CARD_VERSION = "1.0"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_INFERENCE = ("label", "target", "gold", "reference", "text")


@dataclass(frozen=True, slots=True)
class SourceReference:
    citation: str
    url: str
    version_or_revision: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(getattr(self, field), str) or not getattr(self, field).strip()
            for field in self.__dataclass_fields__
        ):
            raise ProvenanceError(
                "source references require explicit citation, URL, and version/revision"
            )

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceReference:
        return cls(**{field: str(data[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class ReproductionProvenanceCard:
    baseline_name: str
    source_paper: SourceReference
    source_repository: SourceReference
    fidelity_summary: str
    deviations: tuple[str, ...]
    allowed_inference_inputs: tuple[str, ...]
    split_plan: ArtifactBinding
    resolved_config: ArtifactBinding
    checkpoint: ArtifactBinding
    architecture_sha256: str
    performance_claims: str = "none"
    schema_version: str = REPRODUCTION_CARD_VERSION

    def __post_init__(self) -> None:
        if not self.baseline_name.strip() or not self.fidelity_summary.strip():
            raise ProvenanceError("baseline_name and fidelity_summary must be explicit")
        if not self.deviations or any(not item.strip() for item in self.deviations):
            raise ProvenanceError("deviations must explicitly state deviations or 'none'")
        if not self.allowed_inference_inputs or any(
            not item.strip() for item in self.allowed_inference_inputs
        ):
            raise ProvenanceError("allowed_inference_inputs must be explicit and non-empty")
        if any(
            any(forbidden in item.casefold() for forbidden in _FORBIDDEN_INFERENCE)
            for item in self.allowed_inference_inputs
        ):
            raise ProvenanceError(
                "allowed_inference_inputs cannot include text labels, targets, or references"
            )
        if not _SHA.fullmatch(self.architecture_sha256):
            raise ProvenanceError("architecture_sha256 must be a lowercase SHA-256 digest")
        if self.performance_claims != "none":
            raise ProvenanceError("reproduction provenance cards cannot contain performance claims")
        if self.schema_version != REPRODUCTION_CARD_VERSION:
            raise ProvenanceError("unsupported reproduction card schema version")

    @property
    def binding_sha256(self) -> str:
        payload = json.dumps(self.binding_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(payload).hexdigest()

    def binding_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_name": self.baseline_name,
            "source_paper": self.source_paper.to_dict(),
            "source_repository": self.source_repository.to_dict(),
            "fidelity_summary": self.fidelity_summary,
            "deviations": list(self.deviations),
            "allowed_inference_inputs": list(self.allowed_inference_inputs),
            "split_plan": self.split_plan.to_dict(),
            "resolved_config": self.resolved_config.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "architecture_sha256": self.architecture_sha256,
            "performance_claims": self.performance_claims,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.binding_dict(), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReproductionProvenanceCard:
        required = set(cls.__dataclass_fields__) | {"binding_sha256"}
        if set(data) != required:
            raise ProvenanceError("reproduction card must contain exactly the required disclosures")
        card = cls(
            baseline_name=str(data["baseline_name"]),
            source_paper=SourceReference.from_dict(data["source_paper"]),
            source_repository=SourceReference.from_dict(data["source_repository"]),
            fidelity_summary=str(data["fidelity_summary"]),
            deviations=tuple(data["deviations"]),
            allowed_inference_inputs=tuple(data["allowed_inference_inputs"]),
            split_plan=ArtifactBinding.from_dict(data["split_plan"]),
            resolved_config=ArtifactBinding.from_dict(data["resolved_config"]),
            checkpoint=ArtifactBinding.from_dict(data["checkpoint"]),
            architecture_sha256=str(data["architecture_sha256"]),
            performance_claims=str(data["performance_claims"]),
            schema_version=str(data["schema_version"]),
        )
        if str(data["binding_sha256"]) != card.binding_sha256:
            raise ProvenanceError("reproduction card binding_sha256 does not match disclosures")
        return card


def write_reproduction_card(path: str | Path, card: ReproductionProvenanceCard) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(card.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_reproduction_card(path: str | Path) -> ReproductionProvenanceCard:
    try:
        return ReproductionProvenanceCard.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ProvenanceError):
            raise
        raise ProvenanceError("invalid reproduction provenance card") from error
