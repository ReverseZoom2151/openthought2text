from __future__ import annotations

import json

from openthought2text.data import (
    DatasetCard,
    SyntheticNeuralTextAdapter,
    audit_dataset_release_bundle,
    build_dataset_release_bundle,
    build_split_plan,
    write_dataset_card,
    write_dataset_release_bundle,
    write_manifest,
)
from openthought2text.data.splits import SplitProtocol

from .test_data_authorized_features import descriptor_for


def release_inputs(root):
    manifest = SyntheticNeuralTextAdapter().generate(str(root))
    manifest_path = root / "manifest.jsonl"
    write_manifest(manifest_path, manifest)
    card_path = root / "dataset_card.json"
    write_dataset_card(card_path, DatasetCard(
        dataset_id=manifest.dataset_id, source="fixture", license="CC0", consent="synthetic",
        access="fixture", modality=("eeg",), splits={"protocol": "random_legacy"},
        preprocessing={"description": "fixture"},
    ))
    plan_path = root / "split_plan.json"
    plan_path.write_text(json.dumps(build_split_plan(manifest.samples, SplitProtocol.RANDOM_LEGACY, seed=2).to_dict()), encoding="utf-8")
    features_path = root / "authorized_features.json"
    features_path.write_text(json.dumps(descriptor_for(manifest, root)), encoding="utf-8")
    return card_path, manifest_path, plan_path, features_path


def test_release_bundle_binds_validated_artifacts_and_information_contract(tmp_path) -> None:
    card, manifest, plan, features = release_inputs(tmp_path)
    bundle = build_dataset_release_bundle(tmp_path, dataset_card=card, canonical_manifest=manifest, derived_split_plan=plan, authorized_feature_descriptor=features)
    bundle_path = tmp_path / "release_bundle.json"
    write_dataset_release_bundle(bundle_path, bundle)

    report = audit_dataset_release_bundle(bundle_path)
    assert report.passed
    assert bundle.information_access.inference_target_text is False
    assert bundle.authorized_feature_descriptor.uri == "authorized_features.json"


def test_release_bundle_audit_detects_changed_bound_artifact(tmp_path) -> None:
    card, manifest, plan, features = release_inputs(tmp_path)
    bundle = build_dataset_release_bundle(tmp_path, dataset_card=card, canonical_manifest=manifest, derived_split_plan=plan, authorized_feature_descriptor=features)
    bundle_path = tmp_path / "release_bundle.json"
    write_dataset_release_bundle(bundle_path, bundle)
    features.write_text("{}", encoding="utf-8")

    assert "ARTIFACT_CHECKSUM_MISMATCH" in {item.code for item in audit_dataset_release_bundle(bundle_path).issues}
