"""Authorization-gated, metadata-first planning for a future ZuCo raw converter.

No MATLAB parser is imported here.  Callers must supply an authorized reader
which returns plain Python mappings; this module validates their alignment
metadata and emits a text-minimized conversion plan only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

ZUCO_RAW_PLAN_KIND = "openthought2text.zuco_raw_conversion_plan"
ZUCO_RAW_PLAN_VERSION = "1.0"


@runtime_checkable
class AuthorizedZuCoReader(Protocol):
    """Boundary protocol owned by an authorized environment, not this package."""

    authorization_id: str

    def read_alignment_records(self, source_root_identifier: str) -> Iterable[Mapping[str, Any]]:
        """Return metadata mappings; implementations may read restricted data elsewhere."""


class ZuCoDataQualitySeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ZuCoDataQualityIssue:
    code: str
    severity: ZuCoDataQualitySeverity
    message: str
    record_index: int | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class ZuCoDataQualityReport:
    record_count: int
    issues: tuple[ZuCoDataQualityIssue, ...] = ()

    @property
    def errors(self) -> tuple[ZuCoDataQualityIssue, ...]:
        return tuple(item for item in self.issues if item.severity == ZuCoDataQualitySeverity.ERROR)

    @property
    def passed(self) -> bool:
        return self.record_count > 0 and not self.errors

    def require_valid(self) -> None:
        if not self.passed:
            codes = ", ".join(item.code for item in self.errors) or "NO_ALIGNMENT_RECORDS"
            raise ValueError(f"ZuCo alignment metadata validation failed: {codes}")


@dataclass(frozen=True, slots=True)
class ZuCoConversionRecord:
    """Text-minimized, canonical planning metadata—not a participant-data payload."""

    subject_id: str
    task: str
    sentence_id: str
    sentence_text_sha256: str
    word_count: int
    fixation_count: int
    recording_id: str
    sampling_rate_hz: float
    channel_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "task": self.task,
            "sentence_id": self.sentence_id,
            "sentence_text_sha256": self.sentence_text_sha256,
            "word_count": self.word_count,
            "fixation_count": self.fixation_count,
            "recording_id": self.recording_id,
            "sampling_rate_hz": self.sampling_rate_hz,
            "channel_count": self.channel_count,
        }


@dataclass(frozen=True, slots=True)
class ZuCoRawConversionPlan:
    authorization_id: str
    source_root_identifier: str
    records: tuple[ZuCoConversionRecord, ...]
    quality_summary: Mapping[str, int]
    version: str = ZUCO_RAW_PLAN_VERSION

    def __post_init__(self) -> None:
        if not self.authorization_id.strip() or not self.source_root_identifier.strip():
            raise ValueError("raw conversion plan needs authorization and source identifiers")
        if not self.records:
            raise ValueError("raw conversion plan requires validated alignment records")
        if self.version != ZUCO_RAW_PLAN_VERSION:
            raise ValueError(f"unsupported ZuCo raw plan version: {self.version!r}")

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": ZUCO_RAW_PLAN_KIND,
            "version": self.version,
            "authorization_id": self.authorization_id,
            "source_root_identifier": self.source_root_identifier,
            "records": [record.to_dict() for record in self.records],
            "quality_summary": dict(self.quality_summary),
            "payload_policy": "metadata_only_no_raw_signals_or_sentence_text",
        }
        if include_checksum:
            encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            data["checksum"] = sha256(encoded.encode("utf-8")).hexdigest()
        return data


def _string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _issue(
    issues: list[ZuCoDataQualityIssue], code: str, message: str, index: int, field: str
) -> None:
    issues.append(ZuCoDataQualityIssue(code, ZuCoDataQualitySeverity.ERROR, message, index, field))


def _validated_record(
    record: Mapping[str, Any], index: int, issues: list[ZuCoDataQualityIssue]
) -> ZuCoConversionRecord | None:
    for key in ("subject_id", "task"):
        if not _string(record.get(key)):
            _issue(issues, "MISSING_RECORD_FIELD", f"{key} must be a non-empty string", index, key)
    sentence, words, fixations, eeg = (
        record.get(key) for key in ("sentence", "words", "fixations", "eeg")
    )
    if not isinstance(sentence, Mapping):
        _issue(
            issues, "MISSING_SENTENCE_ALIGNMENT", "sentence mapping is required", index, "sentence"
        )
    elif not _string(sentence.get("sentence_id")) or not _string(sentence.get("text")):
        _issue(
            issues,
            "MALFORMED_SENTENCE_ALIGNMENT",
            "sentence_id and text are required",
            index,
            "sentence",
        )
    if not isinstance(words, list) or not words:
        _issue(issues, "MISSING_WORD_ALIGNMENT", "words must be a non-empty list", index, "words")
        words = []
    word_indices: set[int] = set()
    for word in words:
        if (
            not isinstance(word, Mapping)
            or not isinstance(word.get("word_index"), int)
            or not _string(word.get("text"))
            or not _finite_positive(word.get("end_s"))
            or not isinstance(word.get("start_s"), (int, float))
            or float(word["start_s"]) < 0
            or float(word["end_s"]) < float(word["start_s"])
        ):
            _issue(
                issues,
                "MALFORMED_WORD_ALIGNMENT",
                "each word needs index, text, and nonnegative timing",
                index,
                "words",
            )
            continue
        word_indices.add(word["word_index"])
    if word_indices and word_indices != set(range(len(words))):
        _issue(
            issues,
            "NONCANONICAL_WORD_INDICES",
            "word indices must be contiguous from zero",
            index,
            "words",
        )
    if not isinstance(fixations, list):
        _issue(issues, "MISSING_FIXATION_ALIGNMENT", "fixations must be a list", index, "fixations")
        fixations = []
    for fixation in fixations:
        if (
            not isinstance(fixation, Mapping)
            or fixation.get("word_index") not in word_indices
            or not isinstance(fixation.get("start_s"), (int, float))
            or not _finite_positive(fixation.get("end_s"))
            or float(fixation["end_s"]) < float(fixation["start_s"])
        ):
            _issue(
                issues,
                "MALFORMED_FIXATION_ALIGNMENT",
                "fixation must reference a word with valid timing",
                index,
                "fixations",
            )
    if not isinstance(eeg, Mapping):
        _issue(issues, "MISSING_EEG_ALIGNMENT", "eeg alignment mapping is required", index, "eeg")
    elif (
        not _string(eeg.get("recording_id"))
        or not _finite_positive(eeg.get("sampling_rate_hz"))
        or not isinstance(eeg.get("channel_count"), int)
        or eeg["channel_count"] < 1
    ):
        _issue(
            issues,
            "MALFORMED_EEG_ALIGNMENT",
            "eeg needs recording_id, positive sampling_rate_hz, and channel_count",
            index,
            "eeg",
        )
    if any(
        item.record_index == index and item.severity == ZuCoDataQualitySeverity.ERROR
        for item in issues
    ):
        return None
    assert isinstance(sentence, Mapping) and isinstance(eeg, Mapping)
    text = str(sentence["text"])
    return ZuCoConversionRecord(
        subject_id=str(record["subject_id"]),
        task=str(record["task"]),
        sentence_id=str(sentence["sentence_id"]),
        sentence_text_sha256=sha256(text.encode("utf-8")).hexdigest(),
        word_count=len(words),
        fixation_count=len(fixations),
        recording_id=str(eeg["recording_id"]),
        sampling_rate_hz=float(eeg["sampling_rate_hz"]),
        channel_count=int(eeg["channel_count"]),
    )


def validate_zuco_alignment_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[ZuCoDataQualityReport, tuple[ZuCoConversionRecord, ...]]:
    """Validate plain reader mappings; no filesystem or MATLAB access occurs here."""
    issues: list[ZuCoDataQualityIssue] = []
    converted: list[ZuCoConversionRecord] = []
    count = 0
    for index, record in enumerate(records):
        count += 1
        if not isinstance(record, Mapping):
            _issue(
                issues,
                "MALFORMED_ALIGNMENT_RECORD",
                "reader record must be a mapping",
                index,
                "record",
            )
            continue
        converted_record = _validated_record(record, index, issues)
        if converted_record is not None:
            converted.append(converted_record)
    if not count:
        issues.append(
            ZuCoDataQualityIssue(
                "NO_ALIGNMENT_RECORDS",
                ZuCoDataQualitySeverity.ERROR,
                "authorized reader returned no records",
            )
        )
    return ZuCoDataQualityReport(count, tuple(issues)), tuple(converted)


def plan_authorized_zuco_raw_conversion(
    reader: AuthorizedZuCoReader,
    *,
    authorization_id: str,
    source_root_identifier: str,
) -> tuple[ZuCoDataQualityReport, ZuCoRawConversionPlan | None]:
    """Ask an explicit authorized reader for metadata and produce a plan if valid."""
    if not isinstance(reader, AuthorizedZuCoReader):
        raise TypeError("reader must implement AuthorizedZuCoReader")
    if not authorization_id.strip() or reader.authorization_id != authorization_id:
        raise PermissionError("authorized reader identifier does not match requested authorization")
    report, records = validate_zuco_alignment_records(
        reader.read_alignment_records(source_root_identifier)
    )
    if not report.passed:
        return report, None
    summary = {
        "records_seen": report.record_count,
        "records_valid": len(records),
        "errors": len(report.errors),
    }
    return report, ZuCoRawConversionPlan(authorization_id, source_root_identifier, records, summary)


def write_zuco_raw_conversion_plan(path: str | Path, plan: ZuCoRawConversionPlan) -> None:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ValueError("ZuCo conversion plans must be written as .json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
