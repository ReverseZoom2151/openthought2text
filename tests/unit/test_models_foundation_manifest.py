from openthought2text.models import (
    FoundationFeatureContract,
    FoundationPretrainingProvenance,
    validate_foundation_checkpoint_manifest,
)


def _args():
    c = FoundationFeatureContract(4, 6)
    p = FoundationPretrainingProvenance("src", "unknown", "desc")
    h = "a" * 64
    keys = ("encoder.weight",)
    m = {
        "file_byte_hash": h,
        "source_name": "src",
        "overlap_label": "unknown",
        "input_feature_size": 4,
        "output_feature_size": 6,
        "frozen_intent": True,
        "license": "MIT",
        "key_schema_summary": ["encoder.weight"],
    }
    return m, h, c, p, keys


def test_manifest_matches_disclosures_without_loading():
    m, h, c, p, k = _args()
    assert validate_foundation_checkpoint_manifest(m, h, c, p, True, k).compatible


def test_manifest_detects_tamper_mismatch_and_missing_disclosures():
    m, h, c, p, k = _args()
    m["file_byte_hash"] = "b" * 64
    r = validate_foundation_checkpoint_manifest(m, h, c, p, True, k)
    assert not r.compatible and any("hash differs" in x for x in r.errors)
    m, h, c, p, k = _args()
    del m["license"]
    m["key_schema_summary"] = ["bad"]
    r = validate_foundation_checkpoint_manifest(m, h, c, p, False, k)
    assert (
        not r.compatible
        and any("license" in x for x in r.errors)
        and any("frozen_intent" in x for x in r.errors)
        and any("not allowed" in x for x in r.errors)
    )
