from __future__ import annotations

import pytest

from openthought2text.data import (
    SplitProtocol,
    audit_authorized_preflight_plan,
    build_authorized_preflight_plan,
    write_authorized_preflight_plan,
)

from .test_data_release_bundle import release_inputs
from openthought2text.data import build_dataset_release_bundle, write_dataset_release_bundle


def inputs(root):
    card, manifest, split, features = release_inputs(root)
    bundle_path = root / "release_bundle.json"
    write_dataset_release_bundle(bundle_path, build_dataset_release_bundle(
        root, dataset_card=card, canonical_manifest=manifest, derived_split_plan=split,
        authorized_feature_descriptor=features,
    ))
    return card, bundle_path, split


def test_preflight_plan_roundtrip_validates_authorized_metadata_only(tmp_path) -> None:
    card, bundle, split = inputs(tmp_path)
    plan = build_authorized_preflight_plan(
        tmp_path, dataset_card=card, release_bundle=bundle, split_plan=split,
        authorization_id="approval-123", source_root_identifier="authorized_fixture_v1",
        requested_protocols=(SplitProtocol.RANDOM_LEGACY,),
    )
    path = tmp_path / "preflight.json"
    write_authorized_preflight_plan(path, plan)
    assert audit_authorized_preflight_plan(path).passed


def test_preflight_rejects_missing_authorization_and_tampered_binding(tmp_path) -> None:
    card, bundle, split = inputs(tmp_path)
    with pytest.raises(ValueError, match="authorization_id"):
        build_authorized_preflight_plan(
            tmp_path, dataset_card=card, release_bundle=bundle, split_plan=split,
            authorization_id="", source_root_identifier="authorized_fixture_v1",
            requested_protocols=(SplitProtocol.RANDOM_LEGACY,),
        )
    plan = build_authorized_preflight_plan(
        tmp_path, dataset_card=card, release_bundle=bundle, split_plan=split,
        authorization_id="approval-123", source_root_identifier="authorized_fixture_v1",
        requested_protocols=(SplitProtocol.RANDOM_LEGACY,),
    )
    path = tmp_path / "preflight.json"
    write_authorized_preflight_plan(path, plan)
    split.write_text("{}", encoding="utf-8")
    assert "ARTIFACT_CHECKSUM_MISMATCH" in {item.code for item in audit_authorized_preflight_plan(path).issues}
