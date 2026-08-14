import pytest

from openthought2text.data import AdapterRegistry, DatasetManifest, InformationAccess

from .test_data_schema import sample


class Adapter:
    name = "demo"

    def build_manifest(self, source):
        return DatasetManifest("zuco", (sample(),), InformationAccess())

    def iter_samples(self, source):
        yield sample()


def test_registry_creates_named_adapter():
    registry = AdapterRegistry()
    registry.register("demo", Adapter)
    assert registry.names() == ("demo",)
    assert next(registry.create("demo").iter_samples("x")).sample_id == "s-1"


def test_registry_rejects_duplicate_name():
    registry = AdapterRegistry()
    registry.register("demo", Adapter)
    with pytest.raises(KeyError):
        registry.register("demo", Adapter)
