"""Strict planning and aggregation artifacts for paired multi-seed benchmarks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from openthought2text.evaluation.records import BenchmarkRowLabel

MULTISEED_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class SeedExpectation:
    seed: int
    prediction_artifact: str
    provenance_artifact: str

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or any(
            not isinstance(getattr(self, field), str) or not getattr(self, field).strip()
            for field in ("prediction_artifact", "provenance_artifact")
        ):
            raise ValueError(
                "seed expectation requires integer seed and explicit artifact references"
            )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class MultiSeedBenchmarkPlan:
    name: str
    label: BenchmarkRowLabel
    expected_metrics: tuple[str, ...]
    seeds: tuple[SeedExpectation, ...]
    no_statistical_claim: str = (
        "Plan and deterministic aggregation only; no statistical claim is made."
    )
    schema_version: str = MULTISEED_VERSION

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.expected_metrics or not self.seeds:
            raise ValueError("name, expected_metrics, and seeds are required")
        if len(set(self.expected_metrics)) != len(self.expected_metrics) or any(
            not metric.strip() for metric in self.expected_metrics
        ):
            raise ValueError("expected metrics must be unique and explicit")
        if (
            len({item.seed for item in self.seeds}) != len(self.seeds)
            or self.schema_version != MULTISEED_VERSION
        ):
            raise ValueError("seeds must be unique and schema version supported")

    @property
    def binding_sha256(self) -> str:
        return _digest(self.to_dict(include_binding=False))

    def to_dict(self, *, include_binding: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "name": self.name,
            "label": self.label.to_dict(),
            "expected_metrics": list(self.expected_metrics),
            "seeds": [item.to_dict() for item in sorted(self.seeds, key=lambda item: item.seed)],
            "no_statistical_claim": self.no_statistical_claim,
        }
        return {**data, "binding_sha256": self.binding_sha256} if include_binding else data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultiSeedBenchmarkPlan:
        plan = cls(
            str(data["name"]),
            BenchmarkRowLabel.from_dict(data["label"]),
            tuple(data["expected_metrics"]),
            tuple(SeedExpectation(**item) for item in data["seeds"]),
            str(data.get("no_statistical_claim", "")),
            str(data["schema_version"]),
        )
        if data.get("binding_sha256") != plan.binding_sha256:
            raise ValueError("multi-seed plan binding_sha256 does not match contents")
        return plan


@dataclass(frozen=True, slots=True)
class SeedMetricResult:
    seed: int
    metrics: Mapping[str, float]
    prediction_artifact: str
    provenance_artifact: str

    def __post_init__(self) -> None:
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("seed metrics must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "prediction_artifact": self.prediction_artifact,
            "provenance_artifact": self.provenance_artifact,
        }


@dataclass(frozen=True, slots=True)
class MultiSeedAggregate:
    plan_binding_sha256: str
    per_seed: tuple[SeedMetricResult, ...]
    mean_metrics: Mapping[str, float]
    no_statistical_claim: str = (
        "Deterministic seed mean only; no confidence interval or statistical claim is made."
    )
    schema_version: str = MULTISEED_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_binding_sha256": self.plan_binding_sha256,
            "per_seed": [
                item.to_dict() for item in sorted(self.per_seed, key=lambda item: item.seed)
            ],
            "mean_metrics": dict(self.mean_metrics),
            "no_statistical_claim": self.no_statistical_claim,
            "binding_sha256": _digest(self._without_binding()),
        }

    def _without_binding(self):
        data = (
            self.to_dict()
            if False
            else {
                "schema_version": self.schema_version,
                "plan_binding_sha256": self.plan_binding_sha256,
                "per_seed": [
                    item.to_dict() for item in sorted(self.per_seed, key=lambda item: item.seed)
                ],
                "mean_metrics": dict(self.mean_metrics),
                "no_statistical_claim": self.no_statistical_claim,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultiSeedAggregate:
        aggregate = cls(
            str(data["plan_binding_sha256"]),
            tuple(SeedMetricResult(**item) for item in data["per_seed"]),
            data["mean_metrics"],
            str(data.get("no_statistical_claim", "")),
            str(data["schema_version"]),
        )
        if data.get("binding_sha256") != _digest(aggregate._without_binding()):
            raise ValueError("multi-seed aggregate binding_sha256 does not match contents")
        return aggregate


def aggregate_multi_seed(
    plan: MultiSeedBenchmarkPlan, results: tuple[SeedMetricResult, ...]
) -> MultiSeedAggregate:
    expected = {item.seed: item for item in plan.seeds}
    supplied = {item.seed: item for item in results}
    if set(expected) != set(supplied) or len(results) != len(supplied):
        raise ValueError("all and only planned unique seeds are required before aggregation")
    for seed, result in supplied.items():
        required = expected[seed]
        if (
            result.prediction_artifact != required.prediction_artifact
            or result.provenance_artifact != required.provenance_artifact
        ):
            raise ValueError("per-seed prediction/provenance artifacts must match the plan")
        if set(result.metrics) != set(plan.expected_metrics):
            raise ValueError("per-seed metric set must match the plan")
    means = {
        metric: sum(float(result.metrics[metric]) for result in results) / len(results)
        for metric in plan.expected_metrics
    }
    return MultiSeedAggregate(
        plan.binding_sha256, tuple(sorted(results, key=lambda item: item.seed)), means
    )


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
