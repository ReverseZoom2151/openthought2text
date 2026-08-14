"""Versioned JSON/JSONL artifacts for predictions and benchmark reports.

Artifacts use only JSON primitives and stable labels so they remain readable after
model code or trainer internals change.  Prediction records are intentionally
separate from reports: one JSONL row per held-out example is auditable without
rewriting a run-level summary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

from openthought2text.controls import ControlCondition


PREDICTION_RECORD_VERSION = "1.0"
EVALUATION_REPORT_VERSION = "1.0"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_scores(scores: Mapping[str, float], name: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for metric, score in scores.items():
        _nonempty(metric, f"{name} metric name")
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError(f"{name}.{metric} must be finite")
        normalized[metric] = numeric
    return normalized


@dataclass(frozen=True, slots=True)
class BenchmarkRowLabel:
    """Canonical label for a result-table row, never a free-form caption.

    The slash form makes it difficult to visually compare an oracle-aligned or
    closed-vocabulary result with a boundary-free, open-vocabulary one.
    """

    dataset: str
    modality: str
    paradigm: str
    alignment: str
    split: str
    vocabulary: str
    decoding: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _nonempty(value, f"benchmark.{name}")
            if "/" in value:
                raise ValueError(f"benchmark.{name} cannot contain '/'")

    @property
    def value(self) -> str:
        return "/".join(getattr(self, name) for name in self.__dataclass_fields__)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkRowLabel":
        return cls(**{name: str(data[name]) for name in cls.__dataclass_fields__})

    @classmethod
    def parse(cls, value: str) -> "BenchmarkRowLabel":
        parts = value.split("/")
        names = tuple(cls.__dataclass_fields__)
        if len(parts) != len(names):
            raise ValueError(f"benchmark label must have {len(names)} slash-separated fields")
        return cls(**dict(zip(names, parts, strict=True)))


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One auditable held-out prediction, serializable as a single JSONL row."""

    sample_id: str
    prediction_text: str
    run_id: str
    control: ControlCondition = ControlCondition.FULL
    reference_text: str | None = None
    target_free: bool = True
    evidence_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PREDICTION_RECORD_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.sample_id, "sample_id")
        _nonempty(self.prediction_text, "prediction_text")
        _nonempty(self.run_id, "run_id")
        _nonempty(self.schema_version, "schema_version")
        object.__setattr__(self, "control", ControlCondition(self.control))
        if self.reference_text is not None:
            _nonempty(self.reference_text, "reference_text")
        if self.evidence_score is not None and not math.isfinite(float(self.evidence_score)):
            raise ValueError("evidence_score must be finite")
        try:
            json.dumps(dict(self.metadata), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must contain JSON-serializable values") from error

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "prediction_text": self.prediction_text,
            "run_id": self.run_id,
            "control": self.control.value,
            "target_free": self.target_free,
            "metadata": dict(self.metadata),
        }
        if self.reference_text is not None:
            data["reference_text"] = self.reference_text
        if self.evidence_score is not None:
            data["evidence_score"] = self.evidence_score
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionRecord":
        return cls(
            sample_id=str(data["sample_id"]),
            prediction_text=str(data["prediction_text"]),
            run_id=str(data["run_id"]),
            control=ControlCondition(data.get("control", ControlCondition.FULL.value)),
            reference_text=data.get("reference_text"),
            target_free=bool(data.get("target_free", True)),
            evidence_score=data.get("evidence_score"),
            metadata=data.get("metadata", {}),
            schema_version=str(data.get("schema_version", PREDICTION_RECORD_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class ControlResult:
    """A metric vector from one control evaluation invocation."""

    condition: ControlCondition
    scores: Mapping[str, float]
    examples: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.examples <= 0:
            raise ValueError("control examples must be positive")
        object.__setattr__(self, "condition", ControlCondition(self.condition))
        object.__setattr__(self, "scores", _finite_scores(self.scores, "control scores"))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "condition": self.condition.value,
            "scores": dict(self.scores),
            "examples": self.examples,
        }
        if self.seed is not None:
            data["seed"] = self.seed
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControlResult":
        return cls(
            condition=ControlCondition(data["condition"]),
            scores=data["scores"],
            examples=int(data["examples"]),
            seed=data.get("seed"),
        )


@dataclass(frozen=True, slots=True)
class ControlAggregate:
    condition: ControlCondition
    runs: int
    examples: int
    mean_scores: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "runs": self.runs,
            "examples": self.examples,
            "mean_scores": dict(self.mean_scores),
        }


def aggregate_control_results(results: Iterable[ControlResult]) -> tuple[ControlAggregate, ...]:
    """Aggregate control metrics, weighted by evaluated example count."""
    grouped: dict[ControlCondition, list[ControlResult]] = defaultdict(list)
    for result in results:
        grouped[result.condition].append(result)
    aggregates: list[ControlAggregate] = []
    for condition in sorted(grouped, key=lambda item: item.value):
        rows = grouped[condition]
        examples = sum(row.examples for row in rows)
        metrics = sorted({metric for row in rows for metric in row.scores})
        means = {
            metric: sum(row.scores[metric] * row.examples for row in rows if metric in row.scores)
            / sum(row.examples for row in rows if metric in row.scores)
            for metric in metrics
        }
        aggregates.append(
            ControlAggregate(condition=condition, runs=len(rows), examples=examples, mean_scores=means)
        )
    return tuple(aggregates)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A versioned run-level report that points to immutable prediction JSONL."""

    run_id: str
    benchmark: BenchmarkRowLabel
    metrics: Mapping[str, float]
    prediction_count: int
    prediction_artifact: str
    control_results: tuple[ControlResult, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVALUATION_REPORT_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _nonempty(self.prediction_artifact, "prediction_artifact")
        _nonempty(self.schema_version, "schema_version")
        if self.prediction_count < 0:
            raise ValueError("prediction_count must be non-negative")
        object.__setattr__(self, "metrics", _finite_scores(self.metrics, "metrics"))
        object.__setattr__(self, "control_results", tuple(self.control_results))
        try:
            json.dumps(dict(self.metadata), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must contain JSON-serializable values") from error

    @property
    def control_aggregates(self) -> tuple[ControlAggregate, ...]:
        return aggregate_control_results(self.control_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "benchmark": self.benchmark.to_dict(),
            "benchmark_label": self.benchmark.value,
            "metrics": dict(self.metrics),
            "prediction_count": self.prediction_count,
            "prediction_artifact": self.prediction_artifact,
            "control_results": [item.to_dict() for item in self.control_results],
            "control_aggregates": [item.to_dict() for item in self.control_aggregates],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationReport":
        return cls(
            run_id=str(data["run_id"]),
            benchmark=BenchmarkRowLabel.from_dict(data["benchmark"]),
            metrics=data["metrics"],
            prediction_count=int(data["prediction_count"]),
            prediction_artifact=str(data["prediction_artifact"]),
            control_results=tuple(ControlResult.from_dict(item) for item in data.get("control_results", ())),
            metadata=data.get("metadata", {}),
            schema_version=str(data.get("schema_version", EVALUATION_REPORT_VERSION)),
        )


def write_prediction_jsonl(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    """Write prediction rows with one complete JSON object per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_prediction_jsonl(path: str | Path) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(PredictionRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid prediction JSONL at line {line_number}") from error
    return tuple(records)


def write_evaluation_report(path: str | Path, report: EvaluationReport) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_evaluation_report(path: str | Path) -> EvaluationReport:
    with Path(path).open(encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError("evaluation report is not valid JSON") from error
    try:
        return EvaluationReport.from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation report violates the result schema") from error
