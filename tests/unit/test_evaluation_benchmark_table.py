import pytest

from openthought2text.evaluation import (
    BenchmarkProvenanceReferences,
    BenchmarkRowLabel,
    BenchmarkTableArtifact,
    BenchmarkTableRow,
    MetricUncertainty,
    render_benchmark_csv,
    render_benchmark_markdown,
)


def _digest(character: str) -> str:
    return character * 64


def _row(dataset: str, run: str) -> BenchmarkTableRow:
    return BenchmarkTableRow(
        BenchmarkRowLabel(dataset, "eeg", "reading", "trial", "loso", "open", "greedy"), run,
        {"wer": 0.2, "retrieval_mrr": 0.7}, {"wer": MetricUncertainty(0.1, 0.3, 0.95)},
        BenchmarkProvenanceReferences("evaluation.json", "provenance.json", _digest("a"), _digest("b")),
    )


def test_table_schema_validates_rows_and_binding_round_trips() -> None:
    table = BenchmarkTableArtifact((_row("zuco_b", "run-b"), _row("zuco_a", "run-a")))
    assert BenchmarkTableArtifact.from_dict(table.to_dict()) == table
    assert [row.label.dataset for row in table.sorted_rows] == ["zuco_a", "zuco_b"]
    with pytest.raises(ValueError, match="labels must be unique"):
        BenchmarkTableArtifact((_row("zuco_a", "one"), _row("zuco_a", "two")))


def test_renderers_are_deterministic_and_include_provenance_and_uncertainty() -> None:
    table = BenchmarkTableArtifact((_row("zuco_b", "run-b"), _row("zuco_a", "run-a")))
    markdown, csv = render_benchmark_markdown(table), render_benchmark_csv(table)
    assert markdown == render_benchmark_markdown(table)
    assert markdown.index("zuco_a/") < markdown.index("zuco_b/")
    assert "[0.1, 0.3] (95%; cluster_bootstrap)" in markdown
    assert _digest("a") in markdown
    assert csv.splitlines()[0].startswith("benchmark_label,run_id,metric,value")
    assert csv.splitlines()[1].startswith("zuco_a/eeg/reading")


def test_row_rejects_unknown_uncertainty_and_malformed_provenance_digest() -> None:
    with pytest.raises(ValueError, match="metrics not present"):
        BenchmarkTableRow(BenchmarkRowLabel("x", "eeg", "read", "trial", "loso", "open", "greedy"), "run", {"wer": 0.2}, {"cer": MetricUncertainty(0.1, 0.3, 0.95)}, BenchmarkProvenanceReferences("e", "p", _digest("a"), _digest("b")))
    with pytest.raises(ValueError, match="SHA-256"):
        BenchmarkProvenanceReferences("e", "p", "bad", _digest("b"))
