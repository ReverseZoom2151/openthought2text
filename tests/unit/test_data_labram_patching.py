from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from openthought2text.data import (
    LaBraMInputPatchConfig,
    TensorBackedSample,
    patch_tensor_backed_sample,
)

from .test_data_schema import sample


def row() -> TensorBackedSample:
    return TensorBackedSample(
        sample=sample(),
        values=torch.tensor([[1.0, 2.0, 3.0, 4.0, 99.0], [5.0, 6.0, 7.0, 8.0, 99.0]]),
        channel_mask=torch.tensor([True, False]),
        time_mask=torch.tensor([True, True, True, True, False]),
    )


def test_patch_contract_segments_deterministically_and_zeroes_padding() -> None:
    config = LaBraMInputPatchConfig(sample_rate_hz=250, patch_size_samples=2)
    patched = patch_tensor_backed_sample(row(), config)
    assert patched.patches.shape == (3, 2, 2)
    assert patched.patch_time_mask.tolist() == [[True, True], [True, True], [False, False]]
    assert patched.patch_mask.tolist() == [True, True, False]
    assert patched.patches[2].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert patched.patches[:, 1, :].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert config.manifest_description()["compatibility_scope"] == "input_geometry_only"


def test_patch_contract_is_label_invariant_and_validates_sample_rate() -> None:
    config = LaBraMInputPatchConfig(sample_rate_hz=250, patch_size_samples=2)
    changed = replace(row(), sample=replace(row().sample, target=None))
    assert torch.equal(
        patch_tensor_backed_sample(row(), config).patches,
        patch_tensor_backed_sample(changed, config).patches,
    )
    wrong_rate = replace(row().sample, signal=replace(row().sample.signal, sampling_rate_hz=200))
    with pytest.raises(ValueError, match="does not match"):
        patch_tensor_backed_sample(replace(row(), sample=wrong_rate), config)
