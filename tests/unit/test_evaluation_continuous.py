import pytest

from openthought2text.evaluation import (
    TimestampedPredictionWindow,
    assemble_continuous_windows,
    summarize_continuous_timing,
)


def _windows():
    return [
        TimestampedPredictionWindow(0.0, 1.0, "alpha", inference_duration_s=0.2, emitted_at_s=1.2),
        TimestampedPredictionWindow(0.5, 1.5, "beta", inference_duration_s=0.3, emitted_at_s=1.8),
    ]


def test_concatenate_policy_preserves_overlap_and_reports_coverage_timing() -> None:
    assembly = assemble_continuous_windows(_windows(), policy="concatenate", require_full_coverage=True)
    assert assembly.merged_text == "alpha beta"
    assert assembly.coverage.source_duration_s == pytest.approx(1.5)
    assert assembly.coverage.overlap_duration_s == pytest.approx(0.5)
    timing = summarize_continuous_timing(assembly)
    assert timing.real_time_factor == pytest.approx(0.5 / 1.5)
    assert timing.mean_latency_s == pytest.approx(0.25)
    assert timing.max_latency_s == pytest.approx(0.3)


def test_drop_overlap_policy_makes_coverage_loss_explicit() -> None:
    assembly = assemble_continuous_windows(_windows(), policy="drop_later_overlap")
    assert len(assembly.selected_windows) == 1
    assert not assembly.coverage.fully_covered
    with pytest.raises(ValueError, match="full source coverage"):
        assemble_continuous_windows(_windows(), policy="drop_later_overlap", require_full_coverage=True)


def test_gap_validation_and_window_contract() -> None:
    windows = [TimestampedPredictionWindow(0.0, 1.0, "one"), TimestampedPredictionWindow(1.2, 2.0, "two")]
    with pytest.raises(ValueError, match="gap"):
        assemble_continuous_windows(windows, max_gap_s=0.1)
    with pytest.raises(ValueError, match="no earlier"):
        TimestampedPredictionWindow(0.0, 1.0, "x", emitted_at_s=0.9)
