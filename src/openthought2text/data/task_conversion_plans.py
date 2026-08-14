"""Authorization-gated metadata planners for Brain2Qwerty and T15.

These contracts accept only plain mappings from injected authorized readers.
They intentionally contain no FIF, EDF, HDF5, or participant-file loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


class TaskConversionSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class TaskConversionIssue:
    code: str
    severity: TaskConversionSeverity
    message: str
    record_index: int | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class TaskConversionQualityReport:
    dataset: str
    record_count: int
    issues: tuple[TaskConversionIssue, ...] = ()

    @property
    def errors(self) -> tuple[TaskConversionIssue, ...]:
        return tuple(item for item in self.issues if item.severity == TaskConversionSeverity.ERROR)

    @property
    def passed(self) -> bool:
        return self.record_count > 0 and not self.errors

    def require_valid(self) -> None:
        if not self.passed:
            raise ValueError(f"{self.dataset} conversion metadata invalid: " + ", ".join(x.code for x in self.errors))


def _string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _issue(issues: list[TaskConversionIssue], code: str, message: str, index: int, field: str) -> None:
    issues.append(TaskConversionIssue(code, TaskConversionSeverity.ERROR, message, index, field))


@runtime_checkable
class AuthorizedBrain2QwertyReader(Protocol):
    authorization_id: str

    def read_typed_event_records(self, source_root_identifier: str) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class Brain2QwertyWindowConfig:
    """Synchronous event window policy; event-oracle use is disclosed, never inferred."""

    duration_s: float = 0.5
    event_oracle_available: bool = False

    def __post_init__(self) -> None:
        if not _positive(self.duration_s):
            raise ValueError("Brain2Qwerty window duration_s must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "window_duration_s": self.duration_s,
            "window_alignment": "synchronous_typed_event",
            "event_oracle_available": self.event_oracle_available,
        }


@dataclass(frozen=True, slots=True)
class Brain2QwertyConversionPlan:
    authorization_id: str
    source_root_identifier: str
    window: Brain2QwertyWindowConfig
    event_count_by_modality: Mapping[str, int]
    event_schema_checksum: str

    def to_dict(self) -> dict[str, object]:
        data = {
            "kind": "openthought2text.brain2qwerty_conversion_plan",
            "authorization_id": self.authorization_id,
            "source_root_identifier": self.source_root_identifier,
            "window": self.window.to_dict(),
            "event_count_by_modality": dict(self.event_count_by_modality),
            "event_schema_checksum": self.event_schema_checksum,
            "payload_policy": "metadata_only_no_raw_signals_or_typed_targets",
        }
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return {**data, "checksum": sha256(encoded.encode("utf-8")).hexdigest()}


def validate_brain2qwerty_typed_events(
    records: Iterable[Mapping[str, Any]], config: Brain2QwertyWindowConfig
) -> TaskConversionQualityReport:
    """Validate typed-event metadata and EEG/MEG separation without signals."""
    issues: list[TaskConversionIssue] = []
    count = 0
    modalities_by_recording: dict[str, str] = {}
    for index, record in enumerate(records):
        count += 1
        if not isinstance(record, Mapping):
            _issue(issues, "MALFORMED_TYPED_EVENT_RECORD", "reader record must be a mapping", index, "record")
            continue
        for key in ("subject_id", "recording_id"):
            if not _string(record.get(key)):
                _issue(issues, "MISSING_TYPED_EVENT_FIELD", f"{key} is required", index, key)
        modality = record.get("modality")
        if modality not in {"eeg", "meg"}:
            _issue(issues, "INVALID_MODALITY", "typed-event modality must be eeg or meg", index, "modality")
        elif _string(record.get("recording_id")):
            prior = modalities_by_recording.setdefault(str(record["recording_id"]), str(modality))
            if prior != modality:
                _issue(issues, "MIXED_MODALITY_RECORDING", "one recording_id cannot mix EEG and MEG events", index, "recording_id")
        event = record.get("event")
        if not isinstance(event, Mapping) or not _string(event.get("event_id")) or not _string(event.get("typed_text")) or not isinstance(event.get("timestamp_s"), (int, float)) or float(event["timestamp_s"]) < 0:
            _issue(issues, "MALFORMED_TYPED_EVENT", "event needs ID, typed_text, and nonnegative timestamp_s", index, "event")
        if not _positive(record.get("sampling_rate_hz")) or not _positive(record.get("recording_duration_s")):
            _issue(issues, "MALFORMED_SIGNAL_TIMELINE", "sampling_rate_hz and recording_duration_s must be positive", index, "signal")
        elif isinstance(event, Mapping) and isinstance(event.get("timestamp_s"), (int, float)) and float(event["timestamp_s"]) + config.duration_s > float(record["recording_duration_s"]):
            _issue(issues, "WINDOW_EXCEEDS_RECORDING", "synchronous 500ms-style window exceeds recording timeline", index, "event.timestamp_s")
    if not count:
        issues.append(TaskConversionIssue("NO_TYPED_EVENTS", TaskConversionSeverity.ERROR, "authorized reader returned no typed events"))
    return TaskConversionQualityReport("brain2qwerty", count, tuple(issues))


def plan_authorized_brain2qwerty_conversion(
    reader: AuthorizedBrain2QwertyReader, *, authorization_id: str, source_root_identifier: str,
    window: Brain2QwertyWindowConfig = Brain2QwertyWindowConfig(),
) -> tuple[TaskConversionQualityReport, Brain2QwertyConversionPlan | None]:
    if not isinstance(reader, AuthorizedBrain2QwertyReader):
        raise TypeError("reader must implement AuthorizedBrain2QwertyReader")
    if not authorization_id.strip() or reader.authorization_id != authorization_id:
        raise PermissionError("authorized reader identifier does not match requested authorization")
    records = tuple(reader.read_typed_event_records(source_root_identifier))
    report = validate_brain2qwerty_typed_events(records, window)
    if not report.passed:
        return report, None
    counts = {modality: sum(record["modality"] == modality for record in records) for modality in ("eeg", "meg")}
    schema = [{key: record[key] for key in ("subject_id", "recording_id", "modality")} for record in records]
    digest = sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return report, Brain2QwertyConversionPlan(authorization_id, source_root_identifier, window, counts, digest)


@runtime_checkable
class AuthorizedT15DescriptorReader(Protocol):
    authorization_id: str

    def read_descriptor_records(self, source_root_identifier: str) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class T15TargetAccessContract:
    train_targets: bool
    validation_targets: bool
    inference_targets: bool

    def __post_init__(self) -> None:
        if not self.train_targets or not self.validation_targets or self.inference_targets:
            raise ValueError("T15 target-access contract requires train/validation only and no inference targets")

    def to_dict(self) -> dict[str, bool]:
        return {"train_targets": self.train_targets, "validation_targets": self.validation_targets, "inference_targets": self.inference_targets}


@dataclass(frozen=True, slots=True)
class T15ConversionPlan:
    authorization_id: str
    source_root_identifier: str
    target_access: T15TargetAccessContract
    block_count: int
    descriptor_schema_checksum: str

    def to_dict(self) -> dict[str, object]:
        data = {
            "kind": "openthought2text.t15_conversion_plan", "authorization_id": self.authorization_id,
            "source_root_identifier": self.source_root_identifier, "target_access": self.target_access.to_dict(),
            "block_count": self.block_count, "descriptor_schema_checksum": self.descriptor_schema_checksum,
            "payload_policy": "descriptor_metadata_only_no_hdf5_loading",
        }
        return {**data, "checksum": sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()}


def validate_t15_descriptor_records(
    records: Iterable[Mapping[str, Any]], target_access: T15TargetAccessContract
) -> TaskConversionQualityReport:
    """Validate authorized block/day/session descriptor mappings without HDF5 access."""
    issues: list[TaskConversionIssue] = []
    count = 0
    seen: set[tuple[str, str, str, str]] = set()
    for index, record in enumerate(records):
        count += 1
        if not isinstance(record, Mapping):
            _issue(issues, "MALFORMED_DESCRIPTOR_RECORD", "descriptor record must be a mapping", index, "record")
            continue
        keys = ("subject_id", "block_id", "day_id", "session_id", "recording_id")
        if any(not _string(record.get(key)) for key in keys):
            _issue(issues, "MISSING_BLOCK_DAY_SESSION_MAPPING", "descriptor needs subject/block/day/session/recording IDs", index, "mapping")
            continue
        identity = tuple(str(record[key]) for key in keys[:4])
        if identity in seen:
            _issue(issues, "DUPLICATE_BLOCK_DAY_SESSION_MAPPING", "block/day/session mapping must be unique per subject", index, "mapping")
        seen.add(identity)
        declared_access = record.get("target_access")
        if declared_access != target_access.to_dict():
            _issue(issues, "TARGET_ACCESS_MISMATCH", "descriptor target_access must match declared contract", index, "target_access")
    if not count:
        issues.append(TaskConversionIssue("NO_DESCRIPTOR_RECORDS", TaskConversionSeverity.ERROR, "authorized reader returned no descriptor records"))
    return TaskConversionQualityReport("t15", count, tuple(issues))


def plan_authorized_t15_conversion(
    reader: AuthorizedT15DescriptorReader, *, authorization_id: str, source_root_identifier: str,
    target_access: T15TargetAccessContract,
) -> tuple[TaskConversionQualityReport, T15ConversionPlan | None]:
    if not isinstance(reader, AuthorizedT15DescriptorReader):
        raise TypeError("reader must implement AuthorizedT15DescriptorReader")
    if not authorization_id.strip() or reader.authorization_id != authorization_id:
        raise PermissionError("authorized reader identifier does not match requested authorization")
    records = tuple(reader.read_descriptor_records(source_root_identifier))
    report = validate_t15_descriptor_records(records, target_access)
    if not report.passed:
        return report, None
    schema = [{key: record[key] for key in ("subject_id", "block_id", "day_id", "session_id", "recording_id")} for record in records]
    digest = sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return report, T15ConversionPlan(authorization_id, source_root_identifier, target_access, len(records), digest)
