import pytest

from openthought2text.data import DatasetManifest, InformationAccess, load_manifest, write_manifest
from openthought2text.data.schema import SchemaError

from .test_data_schema import sample


def test_jsonl_manifest_round_trip(tmp_path):
    manifest = DatasetManifest("zuco", (sample(),), InformationAccess(split_definition="subject"))
    location = tmp_path / "manifest.jsonl"
    write_manifest(location, manifest)
    assert load_manifest(location) == manifest


def test_manifest_rejects_samples_from_another_dataset():
    with pytest.raises(SchemaError, match="dataset_id"):
        DatasetManifest("zuco", (sample(dataset_id="other"),), InformationAccess())
