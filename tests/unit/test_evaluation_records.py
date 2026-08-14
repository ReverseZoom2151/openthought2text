import json

import pytest

from openthought2text.controls import ControlCondition
from openthought2text.evaluation import (
    BenchmarkRowLabel,
    ControlResult,
    EvaluationReport,
    PredictionRecord,
    aggregate_control_results,
    read_evaluation_report,
    read_prediction_jsonl,
    write_evaluation_report,
    write_prediction_jsonl,
)


def _benchmark() -> BenchmarkRowLabel:
    return BenchmarkRowLabel(
        dataset="zuco_v1",
        modality="eeg",
        paradigm="reading",
        alignment="word_aligned",
        split="loso_unique_text",
        vocabulary="open_vocab",
        decoding="greedy_target_free",
    )


def test_benchmark_label_is_structured_and_round_trips() -> None:
    label = _benchmark()
    assert label.value == (
        "zuco_v1/eeg/reading/word_aligned/loso_unique_text/open_vocab/greedy_target_free"
    )
    assert BenchmarkRowLabel.parse(label.value) == label
    with pytest.raises(ValueError, match="7 slash"):
        BenchmarkRowLabel.parse("not/a/full/benchmark/label")


def test_prediction_jsonl_is_versioned_and_round_trips(tmp_path) -> None:
    path = tmp_path / "runs" / "predictions.jsonl"
    rows = [
        PredictionRecord("test-1", "hello", "run-1", reference_text="hello", evidence_score=0.8),
        PredictionRecord("test-2", "there", "run-1", control="shuffled", target_free=True),
    ]
    write_prediction_jsonl(path, rows)
    decoded = read_prediction_jsonl(path)
    assert decoded == tuple(rows)
    payload = json.loads(path.read_text().splitlines()[0])
    assert payload["schema_version"] == "1.0"
    assert payload["control"] == "full"


def test_control_result_aggregation_is_example_weighted() -> None:
    aggregates = aggregate_control_results(
        [
            ControlResult(ControlCondition.SHUFFLED, {"wer": 0.8, "mrr": 0.2}, examples=2, seed=1),
            ControlResult(ControlCondition.SHUFFLED, {"wer": 0.2, "mrr": 0.8}, examples=6, seed=2),
            ControlResult(ControlCondition.ZERO, {"wer": 0.9}, examples=8),
        ]
    )
    shuffled = next(item for item in aggregates if item.condition is ControlCondition.SHUFFLED)
    assert shuffled.runs == 2
    assert shuffled.examples == 8
    assert shuffled.mean_scores == pytest.approx({"wer": 0.35, "mrr": 0.65})


def test_report_serializes_row_label_raw_controls_and_aggregates(tmp_path) -> None:
    report = EvaluationReport(
        run_id="run-1",
        benchmark=_benchmark(),
        metrics={"wer": 0.31, "mrr": 0.48},
        prediction_count=17,
        prediction_artifact="predictions.jsonl",
        control_results=(
            ControlResult("full", {"wer": 0.31}, examples=17),
            ControlResult("shuffled", {"wer": 0.61}, examples=17, seed=3),
        ),
        metadata={"checkpoint": "sha256:example"},
    )
    path = tmp_path / "evaluation.json"
    write_evaluation_report(path, report)
    payload = json.loads(path.read_text())
    assert payload["benchmark_label"] == report.benchmark.value
    assert payload["control_aggregates"][1]["condition"] == "shuffled"
    assert read_evaluation_report(path) == report
