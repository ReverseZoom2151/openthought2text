"""Dependency-light baselines with target-free neural-input inference APIs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    kind: str
    payload: Mapping[str, Any]
    checksum_sha256: str

    @classmethod
    def create(cls, kind: str, payload: Mapping[str, Any]) -> "BaselineArtifact":
        if not kind.strip():
            raise ValueError("artifact kind must be non-empty")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return cls(kind, dict(payload), sha256(canonical.encode("utf-8")).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload), "checksum_sha256": self.checksum_sha256}


class FrequencyTextBaseline:
    """Train-text mode baseline; ``predict(neural_input)`` never accepts targets."""

    def __init__(self) -> None:
        self._prediction: str | None = None

    def fit(self, training_texts: Sequence[str]) -> "FrequencyTextBaseline":
        if not training_texts or any(not isinstance(text, str) or not text.strip() for text in training_texts):
            raise ValueError("training_texts must be non-empty text strings")
        counts = Counter(training_texts)
        self._prediction = min((text for text, count in counts.items() if count == max(counts.values())))
        return self

    def predict(self, neural_input: Sequence[Any]) -> tuple[str, ...]:
        return tuple(self._fitted_prediction() for _ in neural_input)

    def artifact(self) -> BaselineArtifact:
        return BaselineArtifact.create("frequency_text", {"prediction": self._fitted_prediction()})

    def _fitted_prediction(self) -> str:
        if self._prediction is None:
            raise RuntimeError("baseline must be fit before inference")
        return self._prediction


class ConstantTextBaseline:
    """A declared constant-output baseline, optionally fitted to a train-only mode."""

    def __init__(self, constant: str | None = None) -> None:
        if constant is not None and not constant.strip():
            raise ValueError("constant must be non-empty when provided")
        self._constant = constant

    def fit(self, training_texts: Sequence[str]) -> "ConstantTextBaseline":
        if self._constant is None:
            self._constant = FrequencyTextBaseline().fit(training_texts)._fitted_prediction()
        return self

    def predict(self, neural_input: Sequence[Any]) -> tuple[str, ...]:
        if self._constant is None:
            raise RuntimeError("constant baseline must be declared or fit before inference")
        return tuple(self._constant for _ in neural_input)

    def artifact(self) -> BaselineArtifact:
        if self._constant is None:
            raise RuntimeError("constant baseline must be declared or fit before export")
        return BaselineArtifact.create("constant_text", {"constant": self._constant})


class NearestNeighborTextRetrieval:
    """Cosine nearest-neighbor retrieval from neural/text-aligned train embeddings."""

    def __init__(self) -> None:
        self._embeddings: tuple[tuple[float, ...], ...] | None = None
        self._texts: tuple[str, ...] | None = None

    def fit(self, training_neural_embeddings: Sequence[Sequence[float]], training_texts: Sequence[str]) -> "NearestNeighborTextRetrieval":
        embeddings = _matrix(training_neural_embeddings, "training_neural_embeddings")
        if len(embeddings) != len(training_texts) or any(not text.strip() for text in training_texts):
            raise ValueError("training embeddings and text labels must be non-empty and aligned")
        self._embeddings, self._texts = tuple(embeddings), tuple(training_texts)
        return self

    def predict(self, neural_embeddings: Sequence[Sequence[float]]) -> tuple[str, ...]:
        if self._embeddings is None or self._texts is None:
            raise RuntimeError("retrieval baseline must be fit before inference")
        queries = _matrix(neural_embeddings, "neural_embeddings", width=len(self._embeddings[0]))
        output = []
        for query in queries:
            best = max(range(len(self._embeddings)), key=lambda index: (_cosine(query, self._embeddings[index]), -index))
            output.append(self._texts[best])
        return tuple(output)

    def artifact(self) -> BaselineArtifact:
        if self._embeddings is None or self._texts is None:
            raise RuntimeError("retrieval baseline must be fit before export")
        return BaselineArtifact.create("nearest_neighbor_text", {"embeddings": [list(row) for row in self._embeddings], "texts": list(self._texts)})


class RidgeRegressor:
    """Small normal-equation ridge regressor; targets appear only in ``fit``."""

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha, self.fit_intercept = float(alpha), bool(fit_intercept)
        self._weights: tuple[tuple[float, ...], ...] | None = None

    def fit(self, training_features: Sequence[Sequence[float]], training_targets: Sequence[float] | Sequence[Sequence[float]]) -> "RidgeRegressor":
        features = _matrix(training_features, "training_features")
        targets = _target_matrix(training_targets, len(features))
        design = [([1.0] if self.fit_intercept else []) + list(row) for row in features]
        width, outputs = len(design[0]), len(targets[0])
        gram = [[sum(row[i] * row[j] for row in design) + (self.alpha if i == j and (not self.fit_intercept or i != 0) else 0.0) for j in range(width)] for i in range(width)]
        rhs = [[sum(design[row][i] * targets[row][output] for row in range(len(design))) for output in range(outputs)] for i in range(width)]
        self._weights = tuple(tuple(row) for row in _solve(gram, rhs))
        return self

    def predict(self, neural_features: Sequence[Sequence[float]]) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
        if self._weights is None:
            raise RuntimeError("ridge regressor must be fit before inference")
        features = _matrix(neural_features, "neural_features", width=len(self._weights) - int(self.fit_intercept))
        outputs = []
        for row in features:
            design = ([1.0] if self.fit_intercept else []) + list(row)
            outputs.append(tuple(sum(design[i] * self._weights[i][j] for i in range(len(design))) for j in range(len(self._weights[0]))))
        return tuple(item[0] for item in outputs) if len(self._weights[0]) == 1 else tuple(outputs)

    def artifact(self) -> BaselineArtifact:
        if self._weights is None:
            raise RuntimeError("ridge regressor must be fit before export")
        return BaselineArtifact.create("ridge", {"alpha": self.alpha, "fit_intercept": self.fit_intercept, "weights": [list(row) for row in self._weights]})


def _matrix(values: Sequence[Sequence[float]], name: str, width: int | None = None) -> list[tuple[float, ...]]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    rows = [tuple(float(item) for item in row) for row in values]
    expected = len(rows[0]) if width is None else width
    if expected <= 0 or any(len(row) != expected or not all(math.isfinite(item) for item in row) for row in rows):
        raise ValueError(f"{name} must be finite rectangular features")
    return rows


def _target_matrix(values: Sequence[float] | Sequence[Sequence[float]], rows: int) -> list[tuple[float, ...]]:
    if len(values) != rows:
        raise ValueError("training targets must align with training features")
    matrix = [(float(item),) if isinstance(item, (int, float)) else tuple(float(value) for value in item) for item in values]
    if not matrix or any(len(row) != len(matrix[0]) or not all(math.isfinite(item) for item in row) for row in matrix):
        raise ValueError("training targets must be finite and rectangular")
    return matrix


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return -math.inf if denominator == 0 else sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _solve(matrix: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [matrix[row][:] + rhs[row][:] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column], strict=True)]
    return [row[size:] for row in augmented]
