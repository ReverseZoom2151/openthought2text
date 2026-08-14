from openthought2text.controls import ControlCondition
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    ControlResult,
    EvaluationReport,
    GenerationAuditSummary,
    GroundingReport,
    LabelInvarianceResult,
    assess_release_evidence,
)
from openthought2text.reporting import (
    ArtifactBinding,
    InformationAccessContract,
    ModelCardReferenceFailureCode,
    RunArtifactProvenance,
    compute_model_card_reference_bindings,
    generate_model_card,
    validate_model_card_references,
)


def _hash(character: str) -> str:
    return character * 64


def _artifacts():
    bind = lambda name, char: ArtifactBinding(name, f"artifacts/{name}", _hash(char))
    provenance = RunArtifactProvenance(
        "run-1",
        bind("model", "a"),
        bind("checkpoint", "b"),
        bind("manifest", "c"),
        bind("split", "d"),
        bind("config", "e"),
        "abc",
        InformationAccessContract(True, True, False, False, False, False, False, "LOSO", "fixed"),
    )
    evaluation = EvaluationReport(
        "run-1",
        BenchmarkRowLabel("zuco", "eeg", "read", "trial", "loso", "open", "greedy"),
        {"wer": 0.2},
        2,
        "predictions.jsonl",
        tuple(ControlResult(condition, {"wer": 0.5}, 2) for condition in ControlCondition),
        {"wer": GroundingReport(0.2, "noise", 0.5, 0.3, 0.3, False)},
    )
    audit = GenerationAuditSummary((), LabelInvarianceResult(("x",), None, None, False, True))
    return evaluation, provenance, audit


def _references():
    return {"evaluation_report": "evaluation.json", "provenance_report": "provenance.json"}


def test_markdown_reference_validator_accepts_generated_card() -> None:
    evaluation, provenance, audit = _artifacts()
    gate = assess_release_evidence(
        evaluation, provenance, generation_audit=audit, available_controls=tuple(ControlCondition)
    )
    card = generate_model_card(evaluation, provenance, gate, artifact_references=_references())
    result = validate_model_card_references(
        card.markdown,
        compute_model_card_reference_bindings(evaluation, provenance, gate),
        gate_passed=True,
    )
    assert result.valid


def test_validator_rejects_missing_or_tampered_digest() -> None:
    evaluation, provenance, audit = _artifacts()
    gate = assess_release_evidence(
        evaluation, provenance, generation_audit=audit, available_controls=tuple(ControlCondition)
    )
    card = generate_model_card(evaluation, provenance, gate, artifact_references=_references())
    tampered = card.markdown.replace("| Evaluation binding |", "| Missing binding |")
    result = validate_model_card_references(
        tampered,
        compute_model_card_reference_bindings(evaluation, provenance, gate),
        gate_passed=True,
    )
    assert ModelCardReferenceFailureCode.MISSING_BINDING in [
        failure.code for failure in result.failures
    ]


def test_validator_rejects_claimed_status_when_gate_failed() -> None:
    evaluation, provenance, _ = _artifacts()
    failed_gate = assess_release_evidence(evaluation, provenance, generation_audit=None)
    card = generate_model_card(
        evaluation, provenance, failed_gate, artifact_references=_references()
    )
    invalid = card.markdown.replace("**UNSUPPORTED", "**CLAIMED")
    result = validate_model_card_references(
        invalid,
        compute_model_card_reference_bindings(evaluation, provenance, failed_gate),
        gate_passed=False,
    )
    assert ModelCardReferenceFailureCode.CLAIMED_WITH_FAILED_GATE in [
        failure.code for failure in result.failures
    ]
