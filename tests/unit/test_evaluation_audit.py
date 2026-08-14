import pytest

from openthought2text.evaluation import (
    assert_label_invariance,
    assert_target_free_signature,
    audit_label_invariance,
    forbidden_generation_parameters,
)


def test_target_free_generator_rejecting_labels_passes_audit() -> None:
    def generate(neural_input: list[int]) -> list[str]:
        return ["decoded"] * len(neural_input)

    assert_target_free_signature(generate)
    result = assert_label_invariance(generate, [1, 2], ["gold", "text"])
    assert not result.label_argument_accepted
    assert result.invariant


def test_audit_detects_teacher_forcing_style_label_leakage() -> None:
    def unsafe_generate(neural_input: list[int], labels: list[str] | None = None) -> list[str]:
        return labels if labels is not None else ["fallback"] * len(neural_input)

    assert forbidden_generation_parameters(unsafe_generate) == ("labels",)
    with pytest.raises(AssertionError, match="forbidden target"):
        assert_target_free_signature(unsafe_generate)
    result = audit_label_invariance(unsafe_generate, [1], ["gold"], replacement_labels=["wrong"])
    assert result.label_argument_accepted
    assert not result.invariant
    with pytest.raises(AssertionError, match="changed"):
        assert_label_invariance(unsafe_generate, [1], ["gold"], replacement_labels=["wrong"])
