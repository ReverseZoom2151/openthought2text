from __future__ import annotations

import json
from hashlib import sha256

import pytest

from openthought2text.data import (
    AUTHORIZED_FEATURE_KIND,
    AUTHORIZED_FEATURE_VERSION,
    SyntheticNeuralTextAdapter,
    audit_authorized_json_features,
    load_authorized_json_features,
)


def descriptor_for(manifest, root):
    mappings = []
    for sample in manifest.samples:
        path = root / "features" / f"{sample.sample_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps([[1.0, 2.0], [3.0, 4.0]]), encoding="utf-8")
        mappings.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "uri": str(path.relative_to(root)),
                "checksum_sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    header = manifest.header_dict()
    source = {"header": header, "samples": [row.to_dict() for row in manifest.samples]}
    source_checksum = sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    descriptor = {
        "kind": AUTHORIZED_FEATURE_KIND,
        "version": AUTHORIZED_FEATURE_VERSION,
        "authorization": "fixture-authorized",
        "source_manifest_checksum": source_checksum,
        "train_only_audit": {
            "fit_split": "train",
            "fit_sample_ids": [row.sample_id for row in manifest.samples if row.split == "train"],
        },
        "mappings": mappings,
    }
    descriptor["checksum"] = sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return descriptor


def test_authorized_json_feature_loader_maps_canonical_samples_without_raw_parsing(
    tmp_path,
) -> None:
    manifest = SyntheticNeuralTextAdapter().generate(str(tmp_path))
    descriptor = descriptor_for(manifest, tmp_path)

    report = audit_authorized_json_features(manifest, descriptor)
    rows = load_authorized_json_features(manifest, tmp_path, descriptor, split="train")

    assert report.passed
    assert [row.sample.sample_id for row in rows] == ["synthetic-00-000", "synthetic-00-001"]
    assert rows[0].values.shape == (2, 2)


def test_authorized_json_feature_audit_rejects_nontrain_fit_and_manifest_mismatch(tmp_path) -> None:
    manifest = SyntheticNeuralTextAdapter().generate(str(tmp_path))
    descriptor = descriptor_for(manifest, tmp_path)
    descriptor["train_only_audit"]["fit_sample_ids"] = ["synthetic-01-000"]
    descriptor["checksum"] = sha256(
        json.dumps(
            {key: value for key, value in descriptor.items() if key != "checksum"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    report = audit_authorized_json_features(manifest, descriptor)
    assert "NONTRAIN_FIT_SAMPLE" in {issue.code for issue in report.issues}


def test_authorized_json_feature_loader_rejects_tampering_and_nonjson_references(tmp_path) -> None:
    manifest = SyntheticNeuralTextAdapter().generate(str(tmp_path))
    descriptor = descriptor_for(manifest, tmp_path)
    path = tmp_path / descriptor["mappings"][0]["uri"]
    path.write_text("[[1, 2], [3, 4]]", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_authorized_json_features(manifest, tmp_path, descriptor)

    descriptor = descriptor_for(manifest, tmp_path)
    descriptor["mappings"][0]["uri"] = "features/not-json.pt"
    descriptor["checksum"] = sha256(
        json.dumps(
            {key: value for key, value in descriptor.items() if key != "checksum"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="JSON array"):
        load_authorized_json_features(manifest, tmp_path, descriptor)
