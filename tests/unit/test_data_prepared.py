from __future__ import annotations

import pytest
import torch

from openthought2text.data import (
    TensorBackedSample,
    build_prepared_artifact_manifest,
    fit_train_channel_normalizer,
    load_prepared_artifact_manifest,
    write_prepared_artifact_manifest,
)

from .test_data_schema import sample


def prepared(sample_id: str, split: str, values: list[list[float]]) -> TensorBackedSample:
    return TensorBackedSample(
        sample=sample(sample_id=sample_id, split=split),
        values=torch.tensor(values, dtype=torch.float32),
    )


def test_train_only_normalizer_fit_and_apply() -> None:
    train_a = prepared("train-a", "train", [[1.0, 3.0], [2.0, 4.0]])
    train_b = prepared("train-b", "train", [[5.0, 7.0], [6.0, 8.0]])
    normalizer = fit_train_channel_normalizer((train_b, train_a))

    assert normalizer.fit_sample_ids == ("train-a", "train-b")
    combined = torch.cat(
        [normalizer.apply(train_a.values), normalizer.apply(train_b.values)],
        dim=1,
    )
    torch.testing.assert_close(combined.mean(dim=1), torch.zeros(2))
    torch.testing.assert_close(combined.std(dim=1, correction=0), torch.ones(2))
    torch.testing.assert_close(
        normalizer.apply(torch.tensor([[9.0], [10.0]])),
        torch.tensor([[2.2361], [2.2361]]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_normalizer_rejects_non_train_declared_split() -> None:
    with pytest.raises(ValueError, match="non-train"):
        fit_train_channel_normalizer(
            (
                prepared("train", "train", [[1.0], [2.0]]),
                prepared("test", "test", [[3.0], [4.0]]),
            )
        )


def test_prepared_artifact_manifest_is_deterministic_and_checksum_verified(tmp_path) -> None:
    train = prepared("train", "train", [[1.0, 2.0], [3.0, 4.0]])
    test = prepared("test", "test", [[7.0, 8.0], [9.0, 10.0]])
    normalizer = fit_train_channel_normalizer((train,))
    first = build_prepared_artifact_manifest((test, train), normalizer)
    second = build_prepared_artifact_manifest((train, test), normalizer)
    output = tmp_path / "prepared.json"
    write_prepared_artifact_manifest(output, first)

    assert first.checksum == second.checksum
    assert load_prepared_artifact_manifest(output).checksum == first.checksum
    assert [record.sample_id for record in first.records] == ["test", "train"]
