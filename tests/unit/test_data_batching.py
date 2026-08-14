from __future__ import annotations

import torch

from openthought2text.data import (
    TensorBackedSample,
    VariableLengthTensorDataset,
    collate_tensor_backed_samples,
)

from .test_data_schema import sample


def row(
    sample_id: str,
    values: list[list[float]],
    *,
    channel_mask: list[bool] | None = None,
    time_mask: list[bool] | None = None,
) -> TensorBackedSample:
    channels = len(values)
    signal = sample(sample_id=sample_id).signal
    sample_row = sample(
        sample_id=sample_id,
        signal=signal.__class__(
            uri=signal.uri,
            recording_id=f"recording-{sample_id}",
            sampling_rate_hz=signal.sampling_rate_hz,
            channel_count=channels,
        ),
    )
    return TensorBackedSample(
        sample=sample_row,
        values=torch.tensor(values, dtype=torch.float32),
        channel_mask=(
            torch.tensor(channel_mask, dtype=torch.bool) if channel_mask is not None else None
        ),
        time_mask=torch.tensor(time_mask, dtype=torch.bool) if time_mask is not None else None,
    )


def test_variable_length_dataset_and_collator_preserve_ids_and_masks() -> None:
    first = row("first", [[1.0, 2.0], [3.0, 4.0]])
    second = row("second", [[5.0, 6.0, 7.0]], time_mask=[True, False, True])
    dataset = VariableLengthTensorDataset((first, second))
    batch = collate_tensor_backed_samples([dataset[1], dataset[0]])

    assert dataset.sample_ids == ("first", "second")
    assert batch.sample_ids == ("second", "first")
    assert batch.signals.shape == (2, 2, 3)
    assert batch.channel_mask.tolist() == [[True, False], [True, True]]
    assert batch.time_mask.tolist() == [[True, False, True], [True, True, False]]
    assert batch.signals[0, 0].tolist() == [5.0, 0.0, 7.0]
    assert batch.signals[1, 1].tolist() == [3.0, 4.0, 0.0]


def test_collator_is_invariant_to_values_outside_declared_masks() -> None:
    clean = row(
        "masked",
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        channel_mask=[True, False],
        time_mask=[True, False, True],
    )
    changed = row(
        "masked",
        [[1.0, -9999.0, 3.0], [7777.0, 8888.0, 9999.0]],
        channel_mask=[True, False],
        time_mask=[True, False, True],
    )
    first = collate_tensor_backed_samples([clean])
    second = collate_tensor_backed_samples([changed])
    torch.testing.assert_close(first.signals, second.signals)
    torch.testing.assert_close(first.channel_mask, second.channel_mask)
    torch.testing.assert_close(first.time_mask, second.time_mask)


def test_train_normalizer_ignores_masked_padding_values() -> None:
    clean = row("train", [[1.0, 2.0, 3.0]], time_mask=[True, False, True])
    changed = row("train", [[1.0, 9999.0, 3.0]], time_mask=[True, False, True])
    from openthought2text.data import fit_train_channel_normalizer

    torch.testing.assert_close(
        fit_train_channel_normalizer([clean]).mean,
        fit_train_channel_normalizer([changed]).mean,
    )
