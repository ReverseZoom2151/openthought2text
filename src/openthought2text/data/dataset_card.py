"""Strict JSON dataset-card artifacts with required research disclosures."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping


DATASET_CARD_KIND = "openthought2text.dataset_card"
DATASET_CARD_VERSION = "1.0"
_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_mapping(value: object, name: str, required_key: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    result = {str(key): item for key, item in value.items()}
    if any(not key or not isinstance(item, str) or not item.strip() for key, item in result.items()):
        raise ValueError(f"{name} values must be non-empty strings")
    if required_key not in result:
        raise ValueError(f"{name} must disclose {required_key!r}")
    return result


@dataclass(frozen=True, slots=True)
class DatasetCard:
    """Portable disclosure record required before real-data benchmark use."""

    dataset_id: str
    source: str
    license: str
    consent: str
    access: str
    modality: tuple[str, ...]
    splits: Mapping[str, str]
    preprocessing: Mapping[str, str]
    version: str = DATASET_CARD_VERSION

    def __post_init__(self) -> None:
        _nonempty_string(self.dataset_id, "dataset_id")
        _nonempty_string(self.source, "source")
        _nonempty_string(self.license, "license")
        _nonempty_string(self.consent, "consent")
        _nonempty_string(self.access, "access")
        if self.version != DATASET_CARD_VERSION:
            raise ValueError(f"unsupported dataset card version: {self.version!r}")
        if not self.modality or any(not isinstance(item, str) or not item.strip() for item in self.modality):
            raise ValueError("modality must be a non-empty list of strings")
        if len(set(self.modality)) != len(self.modality):
            raise ValueError("modality entries must be unique")
        _string_mapping(self.splits, "splits", "protocol")
        _string_mapping(self.preprocessing, "preprocessing", "description")

    @property
    def checksum(self) -> str:
        return _canonical_hash(self.to_dict(include_checksum=False))

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": DATASET_CARD_KIND,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "source": self.source,
            "license": self.license,
            "consent": self.consent,
            "access": self.access,
            "modality": list(self.modality),
            "splits": dict(self.splits),
            "preprocessing": dict(self.preprocessing),
        }
        if include_checksum:
            data["checksum"] = self.checksum
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetCard":
        if data.get("kind") != DATASET_CARD_KIND:
            raise ValueError("not an OpenThought2Text dataset card")
        try:
            card = cls(
                dataset_id=_nonempty_string(data["dataset_id"], "dataset_id"),
                source=_nonempty_string(data["source"], "source"),
                license=_nonempty_string(data["license"], "license"),
                consent=_nonempty_string(data["consent"], "consent"),
                access=_nonempty_string(data["access"], "access"),
                modality=tuple(data["modality"]),
                splits=_string_mapping(data["splits"], "splits", "protocol"),
                preprocessing=_string_mapping(data["preprocessing"], "preprocessing", "description"),
                version=str(data.get("version", DATASET_CARD_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid dataset card disclosure schema") from error
        expected = data.get("checksum")
        if not isinstance(expected, str) or _CHECKSUM_PATTERN.fullmatch(expected) is None:
            raise ValueError("dataset card needs a lowercase SHA-256 checksum")
        if expected != card.checksum:
            raise ValueError("dataset card checksum does not match its contents")
        return card


@dataclass(frozen=True, slots=True)
class DatasetCardIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class DatasetCardValidationReport:
    path: Path
    card: DatasetCard | None = None
    issues: tuple[DatasetCardIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return self.card is not None and not self.issues

    def require_valid(self) -> DatasetCard:
        if not self.passed:
            codes = ", ".join(issue.code for issue in self.issues) or "invalid dataset card"
            raise ValueError(f"dataset card validation failed: {codes}")
        assert self.card is not None
        return self.card


def validate_dataset_card(path: str | Path) -> DatasetCardValidationReport:
    """Validate a JSON card and return structured errors without YAML parsing."""
    source = Path(path)
    if source.suffix.casefold() != ".json":
        return DatasetCardValidationReport(
            source,
            issues=(DatasetCardIssue("UNSUPPORTED_CARD_FORMAT", "dataset cards must use .json"),),
        )
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        return DatasetCardValidationReport(
            source,
            issues=(DatasetCardIssue("MISSING_CARD_FILE", str(error)),),
        )
    except json.JSONDecodeError as error:
        return DatasetCardValidationReport(
            source,
            issues=(DatasetCardIssue("INVALID_CARD_JSON", str(error)),),
        )
    if not isinstance(data, dict):
        return DatasetCardValidationReport(
            source,
            issues=(DatasetCardIssue("INVALID_CARD_OBJECT", "dataset card must be a JSON object"),),
        )
    try:
        return DatasetCardValidationReport(source, card=DatasetCard.from_dict(data))
    except ValueError as error:
        message = str(error)
        if "checksum" in message:
            code = "INVALID_CARD_CHECKSUM"
        elif "disclosure" in message or "must disclose" in message:
            code = "MISSING_DISCLOSURE"
        elif "not an OpenThought2Text" in message:
            code = "INVALID_CARD_KIND"
        else:
            code = "INVALID_CARD_SCHEMA"
        return DatasetCardValidationReport(source, issues=(DatasetCardIssue(code, message),))


def load_dataset_card(path: str | Path) -> DatasetCard:
    return validate_dataset_card(path).require_valid()


def write_dataset_card(path: str | Path, card: DatasetCard) -> None:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("dataset cards must be written as .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(card.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
