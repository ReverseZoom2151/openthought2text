import pytest

from openthought2text.evaluation import build_grounding_report, grounded_gain


def test_grounded_gain_uses_strongest_higher_is_better_control() -> None:
    gain, name, score = grounded_gain(0.71, {"zero": 0.2, "noise": 0.34, "lm_only": 0.66})
    assert gain == pytest.approx(0.05)
    assert name == "lm_only"
    assert score == pytest.approx(0.66)


def test_grounded_gain_handles_lower_is_better_error_metrics() -> None:
    report = build_grounding_report(
        0.22,
        {"zero": 0.70, "noise": 0.51, "lm_only": 0.32},
        shuffled_score=0.45,
        higher_is_better=False,
    )
    assert report.strongest_control == "lm_only"
    assert report.grounded_gain == pytest.approx(0.10)
    assert report.neural_contribution == pytest.approx(0.23)
