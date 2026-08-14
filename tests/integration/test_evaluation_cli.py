from __future__ import annotations

from openthought2text.cli.main import main
from openthought2text.controls import ControlCondition
from openthought2text.evaluation import PredictionRecord, write_prediction_jsonl


def test_saved_prediction_evaluation_and_report_cli(tmp_path) -> None:
    rows = []
    for condition, prediction in ((ControlCondition.FULL, "hello world"), (ControlCondition.SHUFFLED, "other")):
        rows.append(
            PredictionRecord(
                sample_id="sample-1",
                prediction_text=prediction,
                reference_text="hello world",
                run_id="run-1",
                control=condition,
            )
        )
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "report.json"
    write_prediction_jsonl(predictions, rows)
    benchmark = "synthetic/eeg/reading/trial/subject_disjoint/open_vocab/greedy"
    assert main([
        "evaluate", "saved-predictions", "--predictions", str(predictions), "--benchmark", benchmark,
        "--output", str(output),
    ]) == 0
    assert main(["report", "build", "--report", str(output)]) == 0
    assert main([
        "evaluate", "compare-controls", "--run", str(output), "--controls", "full,shuffled"
    ]) == 0
