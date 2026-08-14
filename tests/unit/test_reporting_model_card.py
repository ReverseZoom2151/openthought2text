import pytest

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
    ModelCardError,
    ModelCardStatus,
    RunArtifactProvenance,
    generate_model_card,
    write_model_card,
)


def _hash(character: str) -> str:
    return character * 64


def _provenance() -> RunArtifactProvenance:
    bind = lambda name, char: ArtifactBinding(name, f"artifacts/{name}", _hash(char))
    return RunArtifactProvenance(
        "run-1",
        bind("conformer", "a"),
        bind("checkpoint", "b"),
        bind("manifest", "c"),
        bind("split", "d"),
        bind("config", "e"),
        "abc123",
        InformationAccessContract(
            True, True, False, False, False, False, False, "LOSO", "fixed windows"
        ),
    )


def _evaluation() -> EvaluationReport:
    return EvaluationReport(
        "run-1",
        BenchmarkRowLabel("zuco", "eeg", "reading", "trial", "loso", "open", "greedy"),
        {"wer": 0.2},
        2,
        "runs/run-1/predictions.jsonl",
        tuple(ControlResult(condition, {"wer": 0.5}, 2) for condition in ControlCondition),
        {"wer": GroundingReport(0.2, "noise", 0.5, 0.3, 0.3, False)},
    )


def _audit() -> GenerationAuditSummary:
    return GenerationAuditSummary((), LabelInvarianceResult(("x",), None, None, False, True))


def _references() -> dict[str, str]:
    return {
        "evaluation_report": "runs/run-1/evaluation.json",
        "provenance_report": "runs/run-1/provenance.json",
    }


def test_claimed_model_card_contains_evidence_and_artifact_references(tmp_path) -> None:
    evaluation, provenance = _evaluation(), _provenance()
    gate = assess_release_evidence(
        evaluation,
        provenance,
        generation_audit=_audit(),
        available_controls=tuple(ControlCondition),
    )
    card = generate_model_card(evaluation, provenance, gate, artifact_references=_references())
    assert card.status is ModelCardStatus.CLAIMED
    assert "**CLAIMED — evidence gate passed.**" in card.markdown
    assert "runs/run-1/evaluation.json" in card.markdown
    assert "`wer`" in card.markdown
    path = tmp_path / "MODEL_CARD.md"
    write_model_card(path, card)
    assert path.read_text() == card.markdown


def test_unsupported_card_names_gate_failures_without_claiming_support() -> None:
    evaluation, provenance = _evaluation(), _provenance()
    gate = assess_release_evidence(evaluation, provenance, generation_audit=None)
    card = generate_model_card(evaluation, provenance, gate, artifact_references=_references())
    assert card.status is ModelCardStatus.UNSUPPORTED
    assert "**UNSUPPORTED" in card.markdown
    assert "`missing_generation_audit`" in card.markdown
    assert "evidence gate passed" not in card.markdown


def test_card_requires_artifact_references_and_matching_run_identity() -> None:
    evaluation, provenance = _evaluation(), _provenance()
    gate = assess_release_evidence(evaluation, provenance, generation_audit=None)
    with pytest.raises(ModelCardError, match="missing required"):
        generate_model_card(evaluation, provenance, gate, artifact_references={})
