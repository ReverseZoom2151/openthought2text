from __future__ import annotations

import json

import pytest

from openthought2text.cli.main import main
from openthought2text.data import SyntheticNeuralTextAdapter, load_manifest


def test_cli_splits_build_writes_derived_manifest_and_sidecar_without_mutating_source(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_manifest = SyntheticNeuralTextAdapter().generate(str(source_root))
    source_path = source_root / "synthetic_manifest.jsonl"
    source_bytes = source_path.read_bytes()
    output = tmp_path / "derived" / "subject_split.jsonl"

    assert main([
        "splits",
        "build",
        "--manifest",
        str(source_path),
        "--output",
        str(output),
        "--protocol",
        "loso_subject",
        "--held-out-subject",
        "synthetic-subject-02",
        "--seed",
        "17",
    ]) == 0

    plan_path = output.with_suffix(".split_plan.json")
    derived = load_manifest(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert source_path.read_bytes() == source_bytes
    assert len(derived.samples) == len(source_manifest.samples)
    assert {sample.subject_id for sample in derived.samples if sample.split == "test"} == {
        "synthetic-subject-02"
    }
    assert plan["protocol"] == "loso_subject"
    assert derived.metadata["split_plan"] == plan


def test_cli_splits_build_refuses_existing_output_or_sidecar(tmp_path) -> None:
    source_root = tmp_path / "source"
    SyntheticNeuralTextAdapter().generate(str(source_root))
    output = tmp_path / "derived.jsonl"
    output.write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([
            "splits",
            "build",
            "--manifest",
            str(source_root / "synthetic_manifest.jsonl"),
            "--output",
            str(output),
            "--protocol",
            "random_legacy",
        ])
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "do not replace\n"
