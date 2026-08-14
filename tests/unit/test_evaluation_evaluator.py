import pytest

from openthought2text.controls import ControlCondition
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    PredictionRecord,
    RetrievalInputs,
    evaluate_saved_predictions,
    read_evaluation_report,
    write_evaluation_report,
    write_prediction_jsonl,
)


def _benchmark() -> BenchmarkRowLabel:
    return BenchmarkRowLabel(
        "zuco_v1", "eeg", "reading", "word_aligned", "loso_unique_text", "open_vocab", "greedy"
    )


def test_saved_record_evaluator_computes_text_retrieval_controls_and_grounding(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        PredictionRecord("a", "yes", "run-1", control="full"),
        PredictionRecord("b", "no", "run-1", control="full"),
        PredictionRecord("a", "no", "run-1", control="shuffled"),
        PredictionRecord("b", "yes", "run-1", control="shuffled"),
        PredictionRecord("a", "no", "run-1", control="zero"),
        PredictionRecord("b", "yes", "run-1", control="zero"),
    ]
    write_prediction_jsonl(path, rows)
    report = evaluate_saved_predictions(
        path,
        benchmark=_benchmark(),
        references={"a": "yes", "b": "no"},
        retrieval_by_control={
            "full": RetrievalInputs([[0.9, 0.1], [0.1, 0.9]], [0, 1]),
            "shuffled": RetrievalInputs([[0.1, 0.9], [0.9, 0.1]], [0, 1]),
        },
    )
    assert report.metrics["cer"] == 0.0
    assert report.metrics["wer"] == 0.0
    assert report.metrics["retrieval_mrr"] == 1.0
    assert report.grounding["wer"].grounded_gain == 1.0
    assert report.grounding["retrieval_mrr"].grounded_gain == 0.5
    assert report.control_aggregates[0].condition is ControlCondition.FULL

    output = tmp_path / "evaluation.json"
    write_evaluation_report(output, report)
    assert read_evaluation_report(output) == report


def test_evaluator_rejects_unpaired_controls_and_missing_references() -> None:
    records = [
        PredictionRecord("a", "yes", "run-1", control="full"),
        PredictionRecord("b", "no", "run-1", control="shuffled"),
    ]
    with pytest.raises(ValueError, match="exactly the full-signal"):
        evaluate_saved_predictions(records, benchmark=_benchmark(), references={"a": "yes", "b": "no"})
    with pytest.raises(ValueError, match="missing reference"):
        evaluate_saved_predictions(
            [PredictionRecord("a", "yes", "run-1", control="full")], benchmark=_benchmark()
        )


def test_evaluator_rejects_conflicting_control_reference_and_misaligned_retrieval() -> None:
    records = [
        PredictionRecord("a", "yes", "run-1", control="full", reference_text="yes"),
        PredictionRecord("a", "no", "run-1", control="shuffled", reference_text="different"),
    ]
    with pytest.raises(ValueError, match="conflicting reference"):
        evaluate_saved_predictions(records, benchmark=_benchmark())

    aligned = [
        PredictionRecord("a", "yes", "run-1", control="full", reference_text="yes"),
        PredictionRecord("a", "no", "run-1", control="shuffled", reference_text="yes"),
    ]
    with pytest.raises(ValueError, match="one row per prediction"):
        evaluate_saved_predictions(
            aligned,
            benchmark=_benchmark(),
            retrieval_by_control={"full": RetrievalInputs([[0.9, 0.1], [0.1, 0.9]], [0, 1])},
        )
