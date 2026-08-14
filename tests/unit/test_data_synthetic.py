from openthought2text.data import (
    AdapterRegistry,
    DatasetAdapter,
    SyntheticNeuralTextAdapter,
    audit_splits,
    load_manifest,
)


def test_synthetic_adapter_is_deterministic_and_implements_protocol(tmp_path) -> None:
    adapter = SyntheticNeuralTextAdapter(seed=19)
    assert isinstance(adapter, DatasetAdapter)
    first = adapter.discover(str(tmp_path))
    second = adapter.discover(str(tmp_path))
    assert first == second
    assert len(first.samples) == 6
    assert audit_splits(first.samples, information_access=first.information_access).passed


def test_synthetic_generate_manifest_validate_and_registry_end_to_end(tmp_path) -> None:
    adapter = SyntheticNeuralTextAdapter()
    generated = adapter.generate(str(tmp_path))
    persisted = load_manifest(tmp_path / "synthetic_manifest.jsonl")
    validation = adapter.validate(str(tmp_path))

    registry = AdapterRegistry()
    registry.register("synthetic", SyntheticNeuralTextAdapter)
    from_registry = registry.create("synthetic").build_manifest(str(tmp_path))

    assert persisted == generated
    assert validation.passed
    assert not validation.missing_signal_files
    assert not validation.invalid_signal_files
    assert from_registry.samples == generated.samples


def test_synthetic_validation_rejects_changed_signal_fixture(tmp_path) -> None:
    adapter = SyntheticNeuralTextAdapter()
    adapter.generate(str(tmp_path))
    (tmp_path / "signals" / "subject-00.json").write_text("[]\n", encoding="utf-8")
    validation = adapter.validate(str(tmp_path))
    assert not validation.passed
    assert validation.invalid_signal_files == (tmp_path / "signals" / "subject-00.json",)
