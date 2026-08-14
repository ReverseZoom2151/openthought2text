import pytest

from openthought2text.controls import ControlCondition
from openthought2text.evaluation import MetricSpec, run_faithfulness_suite


def _exact_match(predictions, references) -> float:
    return float(tuple(predictions) == tuple(references))


def test_suite_runs_controls_and_reports_grounded_gain_and_audit() -> None:
    signal = [[[1.0, 0.0]], [[2.0, 0.0]]]
    references = ("first", "second")

    def generate(neural_input):
        if neural_input == signal:
            return references
        return ("control", "control")

    result = run_faithfulness_suite(
        generate,
        signal,
        references,
        [MetricSpec("exact", _exact_match)],
        controls=("full", "zero", "shuffled", "noise", "mask", "length", "timing"),
        control_context={
            "valid_mask": [[True, True], [True, True]],
            "valid_lengths": [2, 2],
            "event_indices": [[0], [1]],
            "channels": 1,
            "time_steps": 2,
        },
        seed=9,
    )
    assert len(result.conditions) == 7
    assert result.condition("full").scores == {"exact": 1.0}
    assert result.grounding["exact"].grounded_gain == 1.0
    assert result.grounding["exact"].neural_contribution == 1.0
    assert result.audit.passed
    assert not result.audit.label_invariance.label_argument_accepted


def test_suite_flags_unsafe_generation_api_and_label_dependence() -> None:
    def unsafe(neural_input, labels=None):
        return labels if labels is not None else ("base",)

    result = run_faithfulness_suite(
        unsafe,
        [[1.0]],
        ("gold",),
        [MetricSpec("exact", _exact_match)],
        controls=(ControlCondition.FULL, ControlCondition.SHUFFLED, ControlCondition.ZERO),
    )
    assert result.audit.forbidden_parameters == ("labels",)
    assert not result.audit.label_invariance.invariant
    assert not result.audit.passed


def test_suite_requires_full_and_declared_side_information() -> None:
    generator = lambda _: ("x",)
    metric = [MetricSpec("exact", _exact_match)]
    with pytest.raises(ValueError, match="include the full"):
        run_faithfulness_suite(generator, [[1.0]], ("x",), metric, controls=("zero",))
    with pytest.raises(ValueError, match="include the shuffled"):
        run_faithfulness_suite(generator, [[1.0]], ("x",), metric, controls=("full", "zero"))
    with pytest.raises(ValueError, match="valid_mask"):
        run_faithfulness_suite(
            generator, [[1.0]], ("x",), metric, controls=("full", "shuffled", "mask")
        )
