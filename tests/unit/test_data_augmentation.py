from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from openthought2text.data import (
    NeuralAugmentationConfig,
    TensorBackedSample,
    augment_neural_tensor_batch,
    augment_tensor_backed_samples,
    collate_tensor_backed_samples,
)

from .test_data_schema import sample


def row(*, safe_shift=False):
    source = sample()
    if safe_shift:
        source = replace(source, metadata={"time_shift_alignment_safe": True})
    return TensorBackedSample(
        sample=source,
        values=torch.tensor([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]]),
        channel_mask=torch.tensor([True, True]),
        time_mask=torch.tensor([True, True, False]),
    )


def test_augmentation_is_seeded_label_invariant_and_never_changes_masked_values() -> None:
    config = NeuralAugmentationConfig(
        temporal_mask_probability=0.4, channel_dropout_probability=0.4, additive_noise_std=0.2
    )
    first = augment_tensor_backed_samples((row(),), config, seed=9)[0]
    second = augment_tensor_backed_samples((row(),), config, seed=9)[0]

    assert torch.equal(first.values, second.values)
    assert first.sample is row().sample or first.sample.target == row().sample.target
    assert first.values[:, 2].tolist() == [0.0, 0.0]
    assert first.channel_mask.tolist() == [True, True]
    assert first.time_mask.tolist() == [True, True, False]


def test_time_shift_requires_explicit_alignment_safety_and_batch_rejects_it() -> None:
    config = NeuralAugmentationConfig(max_time_shift=1)
    with pytest.raises(ValueError, match="alignment_safe"):
        augment_tensor_backed_samples((row(),), config, seed=1)
    shifted = augment_tensor_backed_samples((row(safe_shift=True),), config, seed=1)[0]
    assert shifted.sample.target == row(safe_shift=True).sample.target
    batch = collate_tensor_backed_samples((row(),))
    with pytest.raises(ValueError, match="before collation"):
        augment_neural_tensor_batch(batch, config, seed=1)


def test_batch_augmentation_preserves_padding_zeroes_and_masks() -> None:
    batch = collate_tensor_backed_samples((row(),))
    augmented = augment_neural_tensor_batch(
        batch, NeuralAugmentationConfig(additive_noise_std=0.5), seed=3
    )
    assert augmented.signals[0, :, 2].tolist() == [0.0, 0.0]
    assert torch.equal(augmented.channel_mask, batch.channel_mask)
    assert torch.equal(augmented.time_mask, batch.time_mask)
