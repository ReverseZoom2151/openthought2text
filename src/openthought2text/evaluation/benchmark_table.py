"""Strict, deterministic benchmark-table artifacts for evidence-backed results."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from typing import Any

from .records import BenchmarkRowLabel

BENCHMARK_TABLE_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MetricUncertainty:
    lower: float
    upper: float
    confidence: float
    unit: str = "cluster_bootstrap"

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(float(item)) for item in (self.lower, self.upper, self.confidence)
        ):
            raise ValueError("uncertainty values must be finite")
        if self.lower > self.upper or not 0 < self.confidence < 1 or not self.unit.strip():
            raise ValueError("uncertainty bounds, confidence, or unit are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricUncertainty:
        return cls(
            float(data["lower"]),
            float(data["upper"]),
            float(data["confidence"]),
            str(data.get("unit", "cluster_bootstrap")),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkProvenanceReferences:
    evaluation_artifact: str
    provenance_artifact: str
    provenance_binding_sha256: str
    release_gate_binding_sha256: str

    def __post_init__(self) -> None:
        for name in ("evaluation_artifact", "provenance_artifact"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("provenance_binding_sha256", "release_gate_binding_sha256"):
            digest = str(getattr(self, name)).casefold()
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            object.__setattr__(self, name, digest)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkProvenanceReferences:
        return cls(**{name: str(data[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class BenchmarkTableRow:
    label: BenchmarkRowLabel
    run_id: str
    metrics: Mapping[str, float]
    uncertainty: Mapping[str, MetricUncertainty]
    provenance: BenchmarkProvenanceReferences

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        metrics = {str(name): float(value) for name, value in self.metrics.items()}
        if not metrics or any(
            not name.strip() or not math.isfinite(value) for name, value in metrics.items()
        ):
            raise ValueError("metrics must be non-empty, named, and finite")
        uncertainty = dict(self.uncertainty)
        unknown = set(uncertainty).difference(metrics)
        if unknown:
            raise ValueError(
                f"uncertainty references metrics not present in the row: {sorted(unknown)}"
            )
        for metric, interval in uncertainty.items():
            if not interval.lower <= metrics[metric] <= interval.upper:
                raise ValueError(f"metric {metric} lies outside its declared uncertainty interval")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "uncertainty", uncertainty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.to_dict(),
            "run_id": self.run_id,
            "metrics": dict(self.metrics),
            "uncertainty": {name: value.to_dict() for name, value in self.uncertainty.items()},
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkTableRow:
        return cls(
            BenchmarkRowLabel.from_dict(data["label"]),
            str(data["run_id"]),
            data["metrics"],
            {
                name: MetricUncertainty.from_dict(value)
                for name, value in data.get("uncertainty", {}).items()
            },
            BenchmarkProvenanceReferences.from_dict(data["provenance"]),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkTableArtifact:
    rows: tuple[BenchmarkTableRow, ...]
    schema_version: str = BENCHMARK_TABLE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_TABLE_VERSION or not self.rows:
            raise ValueError(
                "benchmark table requires a supported schema version and at least one row"
            )
        labels = [row.label.value for row in self.rows]
        if len(labels) != len(set(labels)):
            raise ValueError("benchmark table labels must be unique")
        object.__setattr__(self, "rows", tuple(sorted(self.rows, key=lambda row: row.label.value)))

    @property
    def sorted_rows(self) -> tuple[BenchmarkTableRow, ...]:
        return self.rows

    @property
    def binding_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(include_binding=False), sort_keys=True, separators=(",", ":")
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_binding: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "rows": [row.to_dict() for row in self.sorted_rows],
        }
        if include_binding:
            data["binding_sha256"] = self.binding_sha256
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkTableArtifact:
        table = cls(
            tuple(BenchmarkTableRow.from_dict(row) for row in data["rows"]),
            str(data["schema_version"]),
        )
        if data.get("binding_sha256") != table.binding_sha256:
            raise ValueError("benchmark table binding_sha256 does not match table contents")
        return table


def render_benchmark_markdown(table: BenchmarkTableArtifact) -> str:
    lines = [
        "# Benchmark Table",
        "",
        f"Schema version: `{table.schema_version}`  ",
        f"Binding SHA-256: `{table.binding_sha256}`",
        "",
        "| Benchmark label | Run | Metric | Value | Confidence interval | Evaluation artifact | Provenance artifact | Provenance binding | Gate binding |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in table.sorted_rows:
        for metric in sorted(row.metrics):
            interval = row.uncertainty.get(metric)
            ci = (
                "—"
                if interval is None
                else f"[{interval.lower:.6g}, {interval.upper:.6g}] ({interval.confidence:.0%}; {interval.unit})"
            )
            lines.append(
                f"| {_escape(row.label.value)} | {_escape(row.run_id)} | `{_escape(metric)}` | {row.metrics[metric]:.6g} | {_escape(ci)} | `{_escape(row.provenance.evaluation_artifact)}` | `{_escape(row.provenance.provenance_artifact)}` | `{row.provenance.provenance_binding_sha256}` | `{row.provenance.release_gate_binding_sha256}` |"
            )
    return "\n".join(lines) + "\n"


def render_benchmark_csv(table: BenchmarkTableArtifact) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "benchmark_label",
            "run_id",
            "metric",
            "value",
            "ci_lower",
            "ci_upper",
            "confidence",
            "uncertainty_unit",
            "evaluation_artifact",
            "provenance_artifact",
            "provenance_binding_sha256",
            "release_gate_binding_sha256",
        )
    )
    for row in table.sorted_rows:
        for metric in sorted(row.metrics):
            interval = row.uncertainty.get(metric)
            writer.writerow(
                (
                    row.label.value,
                    row.run_id,
                    metric,
                    _format(row.metrics[metric]),
                    "" if interval is None else _format(interval.lower),
                    "" if interval is None else _format(interval.upper),
                    "" if interval is None else _format(interval.confidence),
                    "" if interval is None else interval.unit,
                    row.provenance.evaluation_artifact,
                    row.provenance.provenance_artifact,
                    row.provenance.provenance_binding_sha256,
                    row.provenance.release_gate_binding_sha256,
                )
            )
    return output.getvalue()


def _escape(value: Any) -> str:
    return (
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    )


def _format(value: float) -> str:
    return f"{float(value):.6g}"
