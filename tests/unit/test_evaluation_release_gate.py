from openthought2text.controls import ControlCondition
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    ControlResult,
    EvaluationReport,
    GateFailureCode,
    GenerationAuditSummary,
    GroundingReport,
    LabelInvarianceResult,
    assess_release_evidence,
)
from openthought2text.reporting import (
    ArtifactBinding,
    InformationAccessContract,
    RunArtifactProvenance,
)


def _hash(character: str) -> str:
    return character * 64


def _provenance(*, unsafe_access: bool = False, run_id: str = "run-1") -> RunArtifactProvenance:
    binding = lambda name, char: ArtifactBinding(name, f"artifacts/{name}", _hash(char))
    return RunArtifactProvenance(
        run_id=run_id,
        model=binding("model", "a"),
        checkpoint=binding("checkpoint", "b"),
        data_manifest=binding("manifest", "c"),
        split_plan=binding("split", "d"),
        config=binding("config", "e"),
        code_revision="abc123",
        information_access=InformationAccessContract(
            True, True, unsafe_access, unsafe_access, False, False, False, "LOSO unique text", "fixed windows"
        ),
    )


def _report(*, controls=None, gain: float = 0.3, contribution: float = 0.2) -> EvaluationReport:
    benchmark = BenchmarkRowLabel("zuco", "eeg", "read", "trial", "loso", "open", "greedy")
    controls = controls or tuple(ControlCondition)
    return EvaluationReport(
        run_id="run-1",
        benchmark=benchmark,
        metrics={"wer": 0.2},
        prediction_count=4,
        prediction_artifact="predictions.jsonl",
        control_results=tuple(ControlResult(control, {"wer": 0.5}, 4) for control in controls),
        grounding={
            "wer": GroundingReport(0.2, "noise", 0.5, contribution, gain, higher_is_better=False)
        },
    )


def _audit(*, safe: bool = True) -> GenerationAuditSummary:
    return GenerationAuditSummary(
        () if safe else ("labels",),
        LabelInvarianceResult(("x",), None, None, False, safe),
    )


def test_complete_evidence_package_passes_gate() -> None:
    result = assess_release_evidence(
        _report(), _provenance(), generation_audit=_audit(), available_controls=tuple(ControlCondition)
    )
    assert result.passed
    assert result.failure_codes == ()


def test_gate_returns_structured_missing_evidence_failures() -> None:
    controls = tuple(condition for condition in ControlCondition if condition is not ControlCondition.TIMING_ONLY)
    result = assess_release_evidence(
        _report(controls=controls, gain=0.0, contribution=0.0),
        _provenance(unsafe_access=True, run_id="other-run"),
        generation_audit=None,
        available_controls=controls,
    )
    assert not result.passed
    assert {
        GateFailureCode.RUN_ID_MISMATCH,
        GateFailureCode.UNSAFE_INFORMATION_ACCESS,
        GateFailureCode.MISSING_GENERATION_AUDIT,
        GateFailureCode.MISSING_DECLARED_CONTROL,
        GateFailureCode.MISSING_REPORTED_CONTROL,
        GateFailureCode.INSUFFICIENT_GROUNDED_GAIN,
        GateFailureCode.INSUFFICIENT_NEURAL_CONTRIBUTION,
    }.issubset(set(result.failure_codes))


def test_gate_flags_failed_target_free_audit_and_can_raise() -> None:
    result = assess_release_evidence(_report(), _provenance(), generation_audit=_audit(safe=False))
    assert GateFailureCode.TARGET_ACCEPTING_GENERATOR in result.failure_codes
    assert GateFailureCode.LABEL_INVARIANCE_FAILURE in result.failure_codes
    try:
        result.require_pass()
    except AssertionError as error:
        assert "target_accepting_generator" in str(error)
    else:
        raise AssertionError("failed release gate should raise")
