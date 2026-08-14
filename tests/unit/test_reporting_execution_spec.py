import pytest

from openthought2text.controls import ControlCondition
from openthought2text.evaluation import BenchmarkRowLabel
from openthought2text.reporting import ArtifactBinding, ProvenanceError, TargetFreeEvaluationSpec


def _hash(char):
    return char * 64


def _spec():
    bind = lambda name, char: ArtifactBinding(name, name + ".json", _hash(char))
    return TargetFreeEvaluationSpec(
        _hash("a"),
        bind("model", "b"),
        bind("checkpoint", "c"),
        bind("config", "d"),
        tuple(ControlCondition),
        (BenchmarkRowLabel("zuco", "eeg", "read", "trial", "loso", "open", "greedy"),),
        ("neural_signal", "sample_mask"),
        ("predictions.jsonl", "evaluation.json"),
    )


def test_execution_spec_roundtrips_and_detects_tampering():
    spec, payload = _spec(), _spec().to_dict()
    assert TargetFreeEvaluationSpec.from_dict(payload) == spec
    payload["required_output_artifacts"] = ["missing.json"]
    with pytest.raises(ProvenanceError, match="binding"):
        TargetFreeEvaluationSpec.from_dict(payload)


def test_execution_spec_rejects_target_input_and_missing_full_control():
    with pytest.raises(ProvenanceError, match="cannot declare"):
        TargetFreeEvaluationSpec(
            _hash("a"),
            _spec().model,
            _spec().checkpoint,
            _spec().resolved_config,
            tuple(ControlCondition),
            _spec().benchmark_rows,
            ("target_text",),
            ("x",),
        )
    with pytest.raises(ProvenanceError, match="include full"):
        TargetFreeEvaluationSpec(
            _hash("a"),
            _spec().model,
            _spec().checkpoint,
            _spec().resolved_config,
            (ControlCondition.ZERO,),
            _spec().benchmark_rows,
            ("neural",),
            ("x",),
        )
