"""Strict preprocessing declarations and caller-supplied array parity checks.

These are metadata contracts, not filtering/epoching implementations; no scipy,
MATLAB, HDF5, or participant-data loader is imported.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PREPROCESSING_RECIPE_VERSION = "1.0"


def _checksum(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BandPowerOperation:
    name: str
    low_hz: float
    high_hz: float
    aggregation: str = "mean_power"

    def __post_init__(self) -> None:
        if not self.name.strip() or not 0 <= self.low_hz < self.high_hz:
            raise ValueError("band-power operation needs name and 0 <= low_hz < high_hz")
        if self.aggregation not in {"mean_power", "log_mean_power"}:
            raise ValueError("band-power aggregation must be mean_power or log_mean_power")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "band_power",
            "name": self.name,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True, slots=True)
class ERPOperation:
    name: str
    epoch_start_s: float
    epoch_end_s: float
    baseline_start_s: float | None = None
    baseline_end_s: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.epoch_end_s <= self.epoch_start_s:
            raise ValueError("ERP operation needs name and epoch_start_s < epoch_end_s")
        if (self.baseline_start_s is None) != (self.baseline_end_s is None):
            raise ValueError("ERP baseline start and end must be supplied together")
        if self.baseline_start_s is not None and self.baseline_end_s <= self.baseline_start_s:
            raise ValueError("ERP baseline_start_s must be below baseline_end_s")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": "erp",
            "name": self.name,
            "epoch_start_s": self.epoch_start_s,
            "epoch_end_s": self.epoch_end_s,
        }
        if self.baseline_start_s is not None:
            value.update(
                {"baseline_start_s": self.baseline_start_s, "baseline_end_s": self.baseline_end_s}
            )
        return value


@dataclass(frozen=True, slots=True)
class PreprocessingRecipe:
    dataset_id: str
    operations: tuple[BandPowerOperation | ERPOperation, ...]
    version: str = PREPROCESSING_RECIPE_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.operations:
            raise ValueError("preprocessing recipe requires dataset_id and operations")
        if self.version != PREPROCESSING_RECIPE_VERSION:
            raise ValueError("unsupported preprocessing recipe version")
        if len({operation.name for operation in self.operations}) != len(self.operations):
            raise ValueError("preprocessing operation names must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "openthought2text.preprocessing_recipe",
            "version": self.version,
            "dataset_id": self.dataset_id,
            "operations": [op.to_dict() for op in self.operations],
        }

    @property
    def checksum(self) -> str:
        return _checksum(self.to_dict())


@dataclass(frozen=True, slots=True)
class PreprocessingFitState:
    recipe_checksum: str
    fit_split: str
    fit_sample_ids: tuple[str, ...]
    learned_parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.recipe_checksum) is None or self.fit_split != "train":
            raise ValueError("preprocessing fit state must bind a recipe and train split only")
        if not self.fit_sample_ids or len(set(self.fit_sample_ids)) != len(self.fit_sample_ids):
            raise ValueError("preprocessing fit state needs unique train sample IDs")
        if any(not math.isfinite(float(value)) for value in self.learned_parameters.values()):
            raise ValueError("preprocessing learned parameters must be finite")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "recipe_checksum": self.recipe_checksum,
            "fit_split": self.fit_split,
            "fit_sample_ids": list(self.fit_sample_ids),
            "learned_parameters": dict(self.learned_parameters),
        }
        return {**value, "checksum": _checksum(value)}


def fit_train_preprocessing_state(
    recipe: PreprocessingRecipe,
    sample_splits: Mapping[str, str],
    *,
    learned_parameters: Mapping[str, float],
) -> PreprocessingFitState:
    if not sample_splits:
        raise ValueError("preprocessing fit needs at least one declared train sample")
    non_train = [sample_id for sample_id, split in sample_splits.items() if split != "train"]
    if non_train:
        raise ValueError(
            "preprocessing fit received non-train samples: " + ", ".join(sorted(non_train))
        )
    return PreprocessingFitState(
        recipe.checksum, "train", tuple(sorted(sample_splits)), dict(learned_parameters)
    )


@dataclass(frozen=True, slots=True)
class PreprocessingArtifactMapping:
    sample_id: str
    split: str
    uri: str
    checksum_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.sample_id.strip()
            or self.split not in {"train", "validation", "test"}
            or not self.uri.strip()
            or _SHA256.fullmatch(self.checksum_sha256) is None
        ):
            raise ValueError("artifact mapping requires sample, canonical split, URI, and SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "uri": self.uri,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True, slots=True)
class CanonicalPreprocessingArtifact:
    dataset_id: str
    recipe: PreprocessingRecipe
    fit_state: PreprocessingFitState
    mappings: tuple[PreprocessingArtifactMapping, ...]

    def __post_init__(self) -> None:
        if (
            self.dataset_id != self.recipe.dataset_id
            or self.fit_state.recipe_checksum != self.recipe.checksum
        ):
            raise ValueError("preprocessing artifact dataset/recipe binding is invalid")
        if not self.mappings or len({mapping.sample_id for mapping in self.mappings}) != len(
            self.mappings
        ):
            raise ValueError("preprocessing artifact needs unique sample mappings")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": "openthought2text.preprocessing_artifact",
            "dataset_id": self.dataset_id,
            "recipe": self.recipe.to_dict(),
            "recipe_checksum": self.recipe.checksum,
            "fit_state": self.fit_state.to_dict(),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
        }
        return {**value, "checksum": _checksum(value)}


@dataclass(frozen=True, slots=True)
class NumericalParityReport:
    shape_match: bool
    finite: bool
    max_absolute_error: float | None
    mean_absolute_error: float | None
    max_relative_error: float | None
    absolute_tolerance: float
    relative_tolerance: float

    @property
    def passed(self) -> bool:
        return bool(
            self.shape_match
            and self.finite
            and self.max_absolute_error is not None
            and self.max_relative_error is not None
            and (
                self.max_absolute_error <= self.absolute_tolerance
                or self.max_relative_error <= self.relative_tolerance
            )
        )


def compare_preprocessing_arrays(
    expected: torch.Tensor | Sequence[float],
    actual: torch.Tensor | Sequence[float],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> NumericalParityReport:
    """Compare caller-provided arrays only; no preprocessing is performed."""
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("parity tolerances must be non-negative")
    left, right = (
        torch.as_tensor(expected, dtype=torch.float64),
        torch.as_tensor(actual, dtype=torch.float64),
    )
    if left.shape != right.shape:
        return NumericalParityReport(
            False, False, None, None, None, absolute_tolerance, relative_tolerance
        )
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return NumericalParityReport(
            True, False, None, None, None, absolute_tolerance, relative_tolerance
        )
    absolute = (left - right).abs()
    relative = absolute / torch.maximum(left.abs(), torch.full_like(left, 1e-12))
    return NumericalParityReport(
        True,
        True,
        float(absolute.max()),
        float(absolute.mean()),
        float(relative.max()),
        absolute_tolerance,
        relative_tolerance,
    )
