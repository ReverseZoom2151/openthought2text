from __future__ import annotations

import pytest

from openthought2text.data import (
    InformationAccess,
    SignalReference,
    build_brain2qwerty_canonical_artifact,
    build_t15_canonical_artifact,
    build_zuco_canonical_artifact,
)

from .test_data_schema import sample


def records():
    source = sample()
    signal = SignalReference("features/s1.json", "r1", 250, 2, checksum_sha256="a" * 64)
    first = {
        "sample_id": "a",
        "source_record_id": "reader:0",
        "subject_id": "s1",
        "signal": signal,
        "interval": source.interval,
        "modality": "eeg",
        "split": "train",
        "target": source.target,
        "task": "reading",
    }
    return first, {
        **first,
        "sample_id": "b",
        "source_record_id": "reader:1",
        "split": "test",
        "target": None,
    }


def access():
    return InformationAccess(
        train_target_text=True,
        validation_target_text=True,
        inference_target_text=False,
        split_definition="subject_holdout",
        alignment_source="authorized_reader",
    )


def plan(kind):
    return {"kind": kind, "checksum": "b" * 64}


def test_authorized_records_build_canonical_artifacts_for_all_three_datasets() -> None:
    first, second = records()
    for builder, kind in (
        (build_zuco_canonical_artifact, "zuco-plan"),
        (build_brain2qwerty_canonical_artifact, "b2q-plan"),
        (build_t15_canonical_artifact, "t15-plan"),
    ):
        artifact = builder((first, second), information_access=access(), source_plan=plan(kind))
        assert artifact.manifest.samples[1].target is None
        assert artifact.provenance_mapping == {"a": "reader:0", "b": "reader:1"}
        assert artifact.to_dict()["manifest_checksum"] == artifact.manifest_checksum


def test_target_policy_and_raw_references_are_rejected() -> None:
    first, second = records()
    with pytest.raises(ValueError, match="must omit target"):
        build_zuco_canonical_artifact(
            (first, {**second, "target": first["target"]}),
            information_access=access(),
            source_plan=plan("zuco"),
        )
    raw = {**first, "signal": SignalReference("raw/results.mat", "r", 250, 2)}
    with pytest.raises(ValueError, match="safe non-raw"):
        build_zuco_canonical_artifact((raw,), information_access=access(), source_plan=plan("zuco"))
