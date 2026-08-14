"""Leakage-safe classical baselines with train-only fit APIs."""

from .classical import (
    BaselineArtifact,
    ConstantTextBaseline,
    FrequencyTextBaseline,
    NearestNeighborTextRetrieval,
    RidgeRegressor,
)

__all__ = [
    "BaselineArtifact",
    "ConstantTextBaseline",
    "FrequencyTextBaseline",
    "NearestNeighborTextRetrieval",
    "RidgeRegressor",
]
