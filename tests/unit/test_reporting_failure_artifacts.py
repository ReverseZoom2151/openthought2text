import json

import pytest

from openthought2text.evaluation import (
    BenchmarkRowLabel,
    ControlResult,
    EvaluationReport,
    PredictionRecord,
    write_prediction_jsonl,
)
from openthought2text.reporting import (
    ArtifactBinding,
    InformationAccessContract,
    ProvenanceError,
    RunArtifactProvenance,
    build_failure_artifact_render,
    load_failure_artifact_render,
    write_provenance_report,
)


def _hash(character):
    return character * 64


def _provenance(run_id="run"):
    bind = lambda name, character: ArtifactBinding(name, name + ".json", _hash(character))
    return RunArtifactProvenance(
        run_id,
        bind("model", "a"),
        bind("checkpoint", "b"),
        bind("data", "c"),
        bind("split", "d"),
        bind("config", "e"),
        "rev",
        InformationAccessContract(True, False, False, False, False, False, False, "loso", "manual"),
    )


def _errors():
    return [
        {
            "sample_id": "s",
            "reference": "good",
            "hypothesis": "bad",
            "category": "substitution",
            "operations": {"substitutions": 1},
        }
    ]


def _records():
    return [
        PredictionRecord("s", "bad", "run", control="full"),
        PredictionRecord("s", "noise", "run", control="noise"),
    ]


def test_pure_failure_payload_renderer_validates_and_escapes_gallery():
    result = build_failure_artifact_render(
        [item.to_dict() for item in _records()],
        _errors(),
        [
            ControlResult("full", {"wer": 1}, 1).to_dict(),
            ControlResult("noise", {"wer": 1}, 1).to_dict(),
        ],
        run_id="run",
        prediction_artifact="predictions.jsonl",
        evaluation_artifact="evaluation.json",
        provenance=_provenance().to_dict(),
    )
    assert result.explorer.cases[0].control_predictions == {"noise": "noise"}
    assert "Failure-case explorer" in result.to_markdown()
    assert "no empirical claim" in result.gallery.html


def test_file_renderer_checks_prediction_reference_and_paired_ids(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"
    error_file, evaluation_file, provenance_file = (
        tmp_path / "errors.json",
        tmp_path / "evaluation.json",
        tmp_path / "provenance.json",
    )
    write_prediction_jsonl(prediction_file, _records())
    error_file.write_text(json.dumps(_errors()), encoding="utf-8")
    evaluation = EvaluationReport(
        "run",
        BenchmarkRowLabel("x", "eeg", "read", "trial", "loso", "open", "greedy"),
        {"wer": 1},
        1,
        str(prediction_file),
        (ControlResult("full", {"wer": 1}, 1), ControlResult("noise", {"wer": 1}, 1)),
    )
    evaluation_file.write_text(json.dumps(evaluation.to_dict()), encoding="utf-8")
    write_provenance_report(provenance_file, _provenance())

    result = load_failure_artifact_render(
        prediction_file, error_file, evaluation_file, provenance_file
    )
    assert result.explorer.provenance_binding_sha256 == _provenance().binding_sha256
    assert "<section>" in result.gallery.html

    payload = evaluation.to_dict()
    payload["prediction_artifact"] = "other.jsonl"
    evaluation_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="prediction_artifact"):
        load_failure_artifact_render(prediction_file, error_file, evaluation_file, provenance_file)


def test_pure_renderer_rejects_unpaired_error_ids():
    errors = _errors()
    errors[0]["sample_id"] = "other"
    with pytest.raises(ValueError, match="match full prediction"):
        build_failure_artifact_render(
            [item.to_dict() for item in _records()],
            errors,
            [
                ControlResult("full", {"wer": 1}, 1).to_dict(),
                ControlResult("noise", {"wer": 1}, 1).to_dict(),
            ],
            run_id="run",
            prediction_artifact="p",
            evaluation_artifact="e",
            provenance=_provenance().to_dict(),
        )


def test_pure_renderer_rejects_error_text_not_bound_to_full_prediction():
    errors = _errors()
    errors[0]["hypothesis"] = "other"
    with pytest.raises(ValueError, match="hypothesis"):
        build_failure_artifact_render(
            [item.to_dict() for item in _records()],
            errors,
            [
                ControlResult("full", {"wer": 1}, 1).to_dict(),
                ControlResult("noise", {"wer": 1}, 1).to_dict(),
            ],
            run_id="run",
            prediction_artifact="p",
            evaluation_artifact="e",
            provenance=_provenance().to_dict(),
        )
