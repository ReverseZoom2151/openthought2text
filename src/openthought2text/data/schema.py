"""Versioned, framework-independent schema for neural-to-text examples.

The schema holds references to neural arrays rather than arrays themselves.
That keeps manifests portable and makes it possible to audit information flow
without loading potentially large recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping
import re


SCHEMA_VERSION = "1.0"


class Modality(str, Enum):
    EEG = "eeg"
    MEG = "meg"
    ECoG = "ecog"
    INTRACORTICAL = "intracortical"
    FMRI = "fmri"
    OTHER = "other"


class SchemaError(ValueError):
    """A manifest record violates the canonical data contract."""


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value


def normalise_text(text: str) -> str:
    """Canonical comparison form used only for split-audit matching."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def text_fingerprint(text: str) -> str:
    return sha256(normalise_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """Half-open time interval in seconds relative to a recording start."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise SchemaError("interval must satisfy 0 <= start_s < end_s")

    def overlaps(self, other: "TimeInterval", tolerance_s: float = 0.0) -> bool:
        if tolerance_s < 0:
            raise ValueError("tolerance_s must be non-negative")
        return self.start_s < other.end_s + tolerance_s and other.start_s < self.end_s + tolerance_s

    def to_dict(self) -> dict[str, float]:
        return {"start_s": self.start_s, "end_s": self.end_s}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimeInterval":
        return cls(start_s=float(data["start_s"]), end_s=float(data["end_s"]))


@dataclass(frozen=True, slots=True)
class SignalReference:
    """How to locate and interpret an array slice without loading it."""

    uri: str
    recording_id: str
    sampling_rate_hz: float
    channel_count: int
    array_key: str | None = None
    checksum_sha256: str | None = None
    channel_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.uri, "signal.uri")
        _nonempty(self.recording_id, "signal.recording_id")
        if self.sampling_rate_hz <= 0:
            raise SchemaError("signal.sampling_rate_hz must be positive")
        if self.channel_count <= 0:
            raise SchemaError("signal.channel_count must be positive")
        if self.channel_names and len(self.channel_names) != self.channel_count:
            raise SchemaError("signal.channel_names must match signal.channel_count")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "uri": self.uri,
            "recording_id": self.recording_id,
            "sampling_rate_hz": self.sampling_rate_hz,
            "channel_count": self.channel_count,
        }
        if self.array_key is not None:
            data["array_key"] = self.array_key
        if self.checksum_sha256 is not None:
            data["checksum_sha256"] = self.checksum_sha256
        if self.channel_names:
            data["channel_names"] = list(self.channel_names)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalReference":
        return cls(
            uri=str(data["uri"]),
            recording_id=str(data["recording_id"]),
            sampling_rate_hz=float(data["sampling_rate_hz"]),
            channel_count=int(data["channel_count"]),
            array_key=data.get("array_key"),
            checksum_sha256=data.get("checksum_sha256"),
            channel_names=tuple(data.get("channel_names", ())),
        )


@dataclass(frozen=True, slots=True)
class TextTarget:
    """A target transcript and optional timing supplied by the dataset."""

    text: str
    language: str = "en"
    token_start_s: tuple[float, ...] = ()
    token_end_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.text, "target.text")
        _nonempty(self.language, "target.language")
        if bool(self.token_start_s) != bool(self.token_end_s):
            raise SchemaError("target token start/end times must be supplied together")
        if self.token_start_s:
            if len(self.token_start_s) != len(self.token_end_s):
                raise SchemaError("target token start/end times have different lengths")
            if any(end < start for start, end in zip(self.token_start_s, self.token_end_s)):
                raise SchemaError("target token end time precedes start time")

    @property
    def fingerprint(self) -> str:
        return text_fingerprint(self.text)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"text": self.text, "language": self.language}
        if self.token_start_s:
            data["token_start_s"] = list(self.token_start_s)
            data["token_end_s"] = list(self.token_end_s)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TextTarget":
        return cls(
            text=str(data["text"]),
            language=str(data.get("language", "en")),
            token_start_s=tuple(float(v) for v in data.get("token_start_s", ())),
            token_end_s=tuple(float(v) for v in data.get("token_end_s", ())),
        )


@dataclass(frozen=True, slots=True)
class InformationAccess:
    """Explicit declaration of information visible to each pipeline stage.

    A valid benchmark must store this beside its examples.  Audits can then
    reject targets, timing, or text context that become visible at inference.
    """

    train_target_text: bool = True
    validation_target_text: bool = True
    inference_target_text: bool = False
    inference_text_context: bool = False
    inference_token_boundaries: bool = False
    inference_event_boundaries: bool = False
    inference_stimulus_audio: bool = False
    split_definition: str = "unknown"
    alignment_source: str = "unknown"

    def __post_init__(self) -> None:
        _nonempty(self.split_definition, "information_access.split_definition")
        _nonempty(self.alignment_source, "information_access.alignment_source")

    @property
    def inference_label_leakage(self) -> bool:
        return self.inference_target_text or self.inference_text_context

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_target_text": self.train_target_text,
            "validation_target_text": self.validation_target_text,
            "inference_target_text": self.inference_target_text,
            "inference_text_context": self.inference_text_context,
            "inference_token_boundaries": self.inference_token_boundaries,
            "inference_event_boundaries": self.inference_event_boundaries,
            "inference_stimulus_audio": self.inference_stimulus_audio,
            "split_definition": self.split_definition,
            "alignment_source": self.alignment_source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InformationAccess":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True, slots=True)
class NeuralTextSample:
    """Canonical manifest row for a neural recording slice and text target."""

    sample_id: str
    dataset_id: str
    subject_id: str
    signal: SignalReference
    interval: TimeInterval
    modality: Modality
    target: TextTarget | None = None
    split: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    trial_id: str | None = None
    group_ids: tuple[str, ...] = ()
    task: str = "unknown"
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "dataset_id", "subject_id", "task"):
            _nonempty(getattr(self, field_name), field_name)
        if self.split is not None:
            _nonempty(self.split, "split")
        if len(set(self.group_ids)) != len(self.group_ids) or any(not group for group in self.group_ids):
            raise SchemaError("group_ids must contain unique non-empty values")

    @property
    def recording_key(self) -> tuple[str, str]:
        return (self.dataset_id, self.signal.recording_id)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "signal": self.signal.to_dict(),
            "interval": self.interval.to_dict(),
            "modality": self.modality.value,
            "task": self.task,
        }
        for name in ("split", "session_id", "run_id", "trial_id"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.target is not None:
            data["target"] = self.target.to_dict()
        if self.group_ids:
            data["group_ids"] = list(self.group_ids)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NeuralTextSample":
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise SchemaError(f"unsupported sample schema version: {version!r}")
        return cls(
            sample_id=str(data["sample_id"]),
            dataset_id=str(data["dataset_id"]),
            subject_id=str(data["subject_id"]),
            signal=SignalReference.from_dict(data["signal"]),
            interval=TimeInterval.from_dict(data["interval"]),
            modality=Modality(data["modality"]),
            target=TextTarget.from_dict(data["target"]) if data.get("target") else None,
            split=data.get("split"),
            session_id=data.get("session_id"),
            run_id=data.get("run_id"),
            trial_id=data.get("trial_id"),
            group_ids=tuple(data.get("group_ids", ())),
            task=str(data.get("task", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )
