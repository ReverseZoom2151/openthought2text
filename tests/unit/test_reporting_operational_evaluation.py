from openthought2text.controls import ControlCondition
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    ControlResult,
    EditOperations,
    PredictionRecord,
    TextErrorCategory,
    TextErrorRecord,
)
from openthought2text.reporting import (
    build_failure_case_explorer,
    validate_complete_control_suite_plan,
)
from openthought2text.reporting.execution_spec import TargetFreeEvaluationSpec
from openthought2text.reporting.provenance import ArtifactBinding


def _hash(c):
    return c * 64


def test_failure_explorer_joins_paired_saved_records():
    records = [
        PredictionRecord("s", "bad", "r", control="full"),
        PredictionRecord("s", "noise", "r", control="noise"),
    ]
    errors = [
        TextErrorRecord(
            "s", "good", "bad", TextErrorCategory.SUBSTITUTION, EditOperations(substitutions=1)
        )
    ]
    artifact = build_failure_case_explorer(
        records,
        errors,
        [
            ControlResult(ControlCondition.FULL, {"wer": 1}, 1),
            ControlResult(ControlCondition.GAUSSIAN_NOISE, {"wer": 1}, 1),
        ],
        prediction_artifact="p",
        evaluation_artifact="e",
        provenance_binding_sha256=_hash("a"),
    )
    assert artifact.cases[0].control_predictions == {"noise": "noise"}


def test_control_suite_validator_requires_all_named_controls_and_outputs():
    bind = lambda n, c: ArtifactBinding(n, n, _hash(c))
    spec = TargetFreeEvaluationSpec(
        _hash("a"),
        bind("m", "b"),
        bind("c", "c"),
        bind("d", "d"),
        (ControlCondition.FULL,),
        (BenchmarkRowLabel("x", "eeg", "read", "trial", "loso", "open", "greedy"),),
        ("neural",),
        ("predictions.jsonl",),
    )
    validation = validate_complete_control_suite_plan(spec)
    assert not validation.valid
    assert "noise" in validation.missing_controls
    assert "evaluation.json" in validation.missing_output_artifacts
