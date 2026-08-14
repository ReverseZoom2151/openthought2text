from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from openthought2text.data import (
    ManifestSplit,
    SyntheticNeuralTextAdapter,
    load_json_tensor_samples,
    select_split_samples,
)


def test_load_json_tensor_samples_from_portable_synthetic_artifact(tmp_path) -> None:
    adapter = SyntheticNeuralTextAdapter()
    manifest = adapter.generate(str(tmp_path))
    loaded = load_json_tensor_samples(manifest, tmp_path)

    assert len(loaded) == len(manifest.samples)
    assert loaded[0].sample.signal.uri == "signals/subject-00.json"
    assert loaded[0].values.shape == (2, 100)
    assert loaded[0].values.dtype == torch.float32
    assert len(load_json_tensor_samples(manifest, tmp_path, split=ManifestSplit.TRAIN)) == 2
    assert [sample.sample_id for sample in select_split_samples(manifest.samples, "val")] == [
        "synthetic-01-000",
        "synthetic-01-001",
    ]


def test_json_signal_loader_rejects_escape_and_invalid_payload(tmp_path) -> None:
    adapter = SyntheticNeuralTextAdapter()
    manifest = adapter.generate(str(tmp_path))
    escaped = replace(
        manifest.samples[0],
        signal=replace(manifest.samples[0].signal, uri="../outside.json", checksum_sha256=None),
    )
    invalid_manifest = replace(manifest, samples=(escaped,) + manifest.samples[1:])
    with pytest.raises(ValueError, match="escapes artifact root"):
        load_json_tensor_samples(invalid_manifest, tmp_path)

    feature_path = tmp_path / "signals" / "subject-00.json"
    feature_path.write_text(json.dumps([[1.0], [2.0], [3.0]]), encoding="utf-8")
    no_checksum = replace(
        manifest.samples[0],
        signal=replace(manifest.samples[0].signal, checksum_sha256=None),
    )
    invalid_manifest = replace(manifest, samples=(no_checksum,) + manifest.samples[1:])
    with pytest.raises(ValueError, match="expected 2 channels"):
        load_json_tensor_samples(invalid_manifest, tmp_path)


def test_json_signal_loader_rejects_checksum_mismatch_and_nonfinite_values(tmp_path) -> None:
    adapter = SyntheticNeuralTextAdapter()
    manifest = adapter.generate(str(tmp_path))
    feature_path = tmp_path / "signals" / "subject-00.json"
    feature_path.write_text(json.dumps([[1.0], [float("nan")]]), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_json_tensor_samples(manifest, tmp_path)

    no_checksum = replace(
        manifest.samples[0],
        signal=replace(manifest.samples[0].signal, checksum_sha256=None),
    )
    invalid_manifest = replace(manifest, samples=(no_checksum,) + manifest.samples[1:])
    with pytest.raises(ValueError, match="non-finite"):
        load_json_tensor_samples(invalid_manifest, tmp_path)
