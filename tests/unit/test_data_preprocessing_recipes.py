from __future__ import annotations

import pytest
import torch

from openthought2text.data.preprocessing import (
    BandPowerOperation,
    CanonicalPreprocessingArtifact,
    ERPOperation,
    PreprocessingArtifactMapping,
    PreprocessingRecipe,
    compare_preprocessing_arrays,
    fit_train_preprocessing_state,
)


def recipe():
    return PreprocessingRecipe(
        "fixture", (BandPowerOperation("alpha", 8, 12), ERPOperation("n400", -0.2, 0.8, -0.2, 0))
    )


def test_recipe_fit_state_and_canonical_artifact_are_train_only_and_checksummed() -> None:
    value = recipe()
    state = fit_train_preprocessing_state(
        value, {"s1": "train", "s2": "train"}, learned_parameters={"scale": 1.0}
    )
    artifact = CanonicalPreprocessingArtifact(
        "fixture",
        value,
        state,
        (PreprocessingArtifactMapping("s1", "train", "features/s1.json", "a" * 64),),
    )
    assert artifact.to_dict()["recipe_checksum"] == value.checksum
    with pytest.raises(ValueError, match="non-train"):
        fit_train_preprocessing_state(value, {"s1": "validation"}, learned_parameters={})


def test_numerical_parity_reports_tolerance_shape_and_nonfinite_conditions() -> None:
    assert compare_preprocessing_arrays(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.000001]), absolute_tolerance=1e-5
    ).passed
    assert not compare_preprocessing_arrays([1.0], [1.0, 2.0]).passed
    assert not compare_preprocessing_arrays([float("nan")], [1.0]).finite
