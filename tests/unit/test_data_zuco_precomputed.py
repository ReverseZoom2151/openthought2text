from __future__ import annotations

from hashlib import sha256

from openthought2text.data import (
    DatasetAdapter,
    DatasetManifest,
    InformationAccess,
    SignalReference,
    ZuCoPrecomputedFeatureAdapter,
    write_manifest,
)

from .test_data_schema import sample


def write_feature_artifact(root, *, metadata=None, checksum=None, uri="features/s-1.pt"):
    feature_path = root / "features" / "s-1.pt"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_bytes(b"portable tensor bytes; adapter never deserializes them")
    actual_checksum = sha256(feature_path.read_bytes()).hexdigest()
    signal = SignalReference(
        uri=uri,
        recording_id="feature-recording",
        sampling_rate_hz=100.0,
        channel_count=2,
        checksum_sha256=actual_checksum if checksum is None else checksum,
    )
    default_metadata = {
        "artifact_type": "zuco_precomputed_features",
        "authorization": "fixture-authorized",
        "feature_storage": "torch_pt",
        "feature_layout": "channels_time",
        "alignment_regime": "word_aligned",
        "preprocessing_version": "fixture-1",
    }
    if metadata is not None:
        default_metadata.update(metadata)
    manifest = DatasetManifest(
        dataset_id="zuco_v1_features",
        samples=(sample(dataset_id="zuco_v1_features", signal=signal),),
        information_access=InformationAccess(
            split_definition="subject_disjoint",
            alignment_source="eye_tracking_fixation",
        ),
        metadata=default_metadata,
    )
    write_manifest(root / "manifest.jsonl", manifest)
    return manifest, feature_path


def test_precomputed_zuco_adapter_validates_portable_artifact_without_tensor_load(tmp_path) -> None:
    manifest, _ = write_feature_artifact(tmp_path)
    adapter = ZuCoPrecomputedFeatureAdapter()
    report = adapter.discover(str(tmp_path))

    assert isinstance(adapter, DatasetAdapter)
    assert report.passed
    assert adapter.build_manifest(str(tmp_path)) == manifest
    assert [row.sample_id for row in adapter.iter_samples(str(tmp_path))] == ["s-1"]


def test_precomputed_zuco_adapter_reports_missing_metadata_and_feature_file(tmp_path) -> None:
    write_feature_artifact(tmp_path, metadata={"alignment_regime": ""})
    (tmp_path / "features" / "s-1.pt").unlink()
    report = ZuCoPrecomputedFeatureAdapter().validate(str(tmp_path))

    codes = {issue.code for issue in report.errors}
    assert {"MISSING_FEATURE_METADATA", "MISSING_FEATURE_FILE"} <= codes
    assert not report.passed


def test_precomputed_zuco_adapter_detects_checksum_and_nonportable_reference(tmp_path) -> None:
    write_feature_artifact(tmp_path, checksum="0" * 64, uri="/outside/artifact.pt")
    report = ZuCoPrecomputedFeatureAdapter().discover(str(tmp_path))

    assert "NONPORTABLE_FEATURE_REFERENCE" in {issue.code for issue in report.errors}
    assert not report.passed


def test_precomputed_zuco_adapter_rejects_changed_feature_bytes(tmp_path) -> None:
    _, feature_path = write_feature_artifact(tmp_path)
    feature_path.write_bytes(b"changed")
    report = ZuCoPrecomputedFeatureAdapter().discover(str(tmp_path))

    assert "FEATURE_CHECKSUM_MISMATCH" in {issue.code for issue in report.errors}
