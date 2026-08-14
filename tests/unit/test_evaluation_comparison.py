import pytest

from openthought2text.evaluation import (
    BenchmarkRowLabel,
    MetricDirection,
    NamedBenchmarkResult,
    compare_benchmark_results,
    render_comparison_markdown,
)


def _result(name, metrics):
    return NamedBenchmarkResult(
        name, BenchmarkRowLabel("zuco", "eeg", "read", "trial", "loso", "open", "greedy"), metrics,
        {"sample-a": "abc", "sample-b": "def"},
    )


def test_paired_comparison_is_directional_deterministic_and_not_statistical() -> None:
    comparison = compare_benchmark_results(
        _result("full", {"wer": 0.2, "mrr": 0.5}), _result("shuffled", {"wer": 0.4, "mrr": 0.3}),
        {"wer": "lower_is_better", "mrr": MetricDirection.HIGHER_IS_BETTER},
    )
    assert [(item.metric, item.directional_delta) for item in comparison.deltas] == [("mrr", -0.2), ("wer", -0.2)]
    markdown = render_comparison_markdown(comparison)
    assert "no statistical significance claim" in markdown
    assert markdown == render_comparison_markdown(comparison)


def test_comparison_rejects_unpaired_labels_references_or_metrics() -> None:
    baseline = _result("a", {"wer": 0.2})
    wrong_references = NamedBenchmarkResult("b", baseline.label, {"wer": 0.3}, {"sample-a": "different"})
    with pytest.raises(ValueError, match="fingerprints"):
        compare_benchmark_results(baseline, wrong_references, {"wer": "lower_is_better"})
    with pytest.raises(ValueError, match="exactly the same metrics"):
        compare_benchmark_results(baseline, _result("b", {"wer": 0.3}), {})
