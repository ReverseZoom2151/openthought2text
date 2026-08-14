import pytest

from openthought2text.evaluation import calibration_summary, risk_coverage_curve


def test_calibration_summary_has_expected_ece_brier_and_empty_bins() -> None:
    report = calibration_summary([0.9, 0.8, 0.1, 0.2], [1, 0, 0, 0], bins=2)
    assert report.expected_calibration_error == pytest.approx(0.25)
    assert report.brier_score == pytest.approx(0.175)
    assert [(item.count, item.empirical_accuracy) for item in report.bins] == [(2, 0.0), (2, 0.5)]


def test_risk_coverage_orders_by_confidence_and_reports_withholding_curve() -> None:
    curve = risk_coverage_curve([0.9, 0.8, 0.1, 0.2], [1, 0, 0, 0])
    assert [point.coverage for point in curve] == [0.25, 0.5, 0.75, 1.0]
    assert [point.risk for point in curve] == pytest.approx([0.0, 0.5, 2 / 3, 0.75])
    assert [point.minimum_confidence for point in curve] == [0.9, 0.8, 0.2, 0.1]


def test_calibration_rejects_invalid_numeric_contracts() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        calibration_summary([1.2], [1])
    with pytest.raises(ValueError, match="binary"):
        risk_coverage_curve([0.5], [2])
    with pytest.raises(ValueError, match="equally sized"):
        calibration_summary([0.5], [])
