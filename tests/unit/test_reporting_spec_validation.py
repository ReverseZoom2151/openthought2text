import json

import pytest

from openthought2text.controls import ControlCondition
from openthought2text.evaluation import BenchmarkRowLabel
from openthought2text.reporting import (
    ArtifactBinding,
    ProvenanceError,
    TargetFreeEvaluationSpec,
    validate_serialized_target_free_spec,
    validate_target_free_spec_file,
)


def _hash(character: str) -> str:
    return character * 64


def _payload(*, outputs=("predictions.jsonl", "evaluation.json", "provenance.json")):
    bind = lambda name, character: ArtifactBinding(name, f"{name}.json", _hash(character))
    return TargetFreeEvaluationSpec(
        _hash("a"),
        bind("model", "b"),
        bind("checkpoint", "c"),
        bind("config", "d"),
        tuple(ControlCondition),
        (BenchmarkRowLabel("zuco", "eeg", "read", "trial", "loso", "open", "greedy"),),
        ("neural_signal", "sample_mask"),
        outputs,
    ).to_dict()


def test_serialized_spec_validation_is_pure_and_renders_complete_control_plan():
    payload = _payload()
    result = validate_serialized_target_free_spec(payload)

    assert result.source_path is None
    assert result.valid
    assert result.execution_spec_binding_sha256 == payload["binding_sha256"]
    assert "PASS" in result.to_markdown()
    assert (
        result.to_dict()["no_performance_claim"]
        == "Plan validation only; no evaluation was executed."
    )


def test_file_validation_loads_only_plan_and_reports_missing_required_outputs(tmp_path):
    path = tmp_path / "execution-spec.json"
    path.write_text(json.dumps(_payload(outputs=("predictions.jsonl",))), encoding="utf-8")

    result = validate_target_free_spec_file(path)

    assert result.source_path == str(path)
    assert not result.valid
    assert result.control_suite.missing_output_artifacts == ("evaluation.json", "provenance.json")
    assert "Missing outputs: evaluation.json, provenance.json" in result.to_markdown()


def test_file_validation_rejects_tampered_spec(tmp_path):
    path = tmp_path / "tampered.json"
    payload = _payload()
    payload["inference_fields"] = ["target_text"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="cannot declare"):
        validate_target_free_spec_file(path)
