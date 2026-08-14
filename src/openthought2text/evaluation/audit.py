"""Audits for target-free neural-to-text generation.

These utilities are intentionally model-framework agnostic.  A caller wraps the
production inference method in a one-argument callable, then this module verifies
that target labels cannot alter the emitted prediction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from typing import Any


_FORBIDDEN_TARGET_NAMES = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "target_ids",
        "target_ids_batch",
        "decoder_input_ids",
        "gold_text",
        "reference_text",
    }
)


def forbidden_generation_parameters(function: Callable[..., Any]) -> tuple[str, ...]:
    """Return explicit generation parameters that could carry gold targets."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return ()
    return tuple(
        parameter.name
        for parameter in parameters
        if parameter.name.casefold() in _FORBIDDEN_TARGET_NAMES
    )


def assert_target_free_signature(function: Callable[..., Any]) -> None:
    forbidden = forbidden_generation_parameters(function)
    if forbidden:
        joined = ", ".join(forbidden)
        raise AssertionError(f"generation API accepts forbidden target field(s): {joined}")


def _freeze(value: Any) -> Any:
    """Turn common tensor/array results into equality-safe nested values."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class LabelInvarianceResult:
    baseline_prediction: Any
    label_prediction: Any | None
    replacement_prediction: Any | None
    label_argument_accepted: bool
    invariant: bool


def audit_label_invariance(
    generator: Callable[..., Any],
    neural_input: Any,
    labels: Any,
    *,
    replacement_labels: Any | None = None,
    target_keyword: str = "labels",
    generator_kwargs: Mapping[str, Any] | None = None,
) -> LabelInvarianceResult:
    """Verify predictions stay fixed when labels are omitted, supplied, or replaced.

    A strict production generator should reject the label argument entirely.  The
    result records that distinction instead of treating rejection as a failure.
    """
    kwargs = dict(generator_kwargs or {})
    baseline = _freeze(generator(neural_input, **kwargs))
    if replacement_labels is None:
        replacement_labels = _replacement(labels)
    try:
        label_prediction = _freeze(generator(neural_input, **kwargs, **{target_keyword: labels}))
        replacement_prediction = _freeze(
            generator(neural_input, **kwargs, **{target_keyword: replacement_labels})
        )
    except TypeError:
        return LabelInvarianceResult(
            baseline_prediction=baseline,
            label_prediction=None,
            replacement_prediction=None,
            label_argument_accepted=False,
            invariant=True,
        )
    return LabelInvarianceResult(
        baseline_prediction=baseline,
        label_prediction=label_prediction,
        replacement_prediction=replacement_prediction,
        label_argument_accepted=True,
        invariant=(baseline == label_prediction == replacement_prediction),
    )


def assert_label_invariance(*args: Any, **kwargs: Any) -> LabelInvarianceResult:
    result = audit_label_invariance(*args, **kwargs)
    if not result.invariant:
        raise AssertionError("generation changed when target labels were supplied or replaced")
    return result


def _replacement(labels: Any) -> Any:
    if isinstance(labels, str):
        return "[target-free audit replacement]"
    if hasattr(labels, "flip"):
        try:
            return labels.flip(0)
        except (TypeError, RuntimeError):
            pass
    if isinstance(labels, tuple):
        return tuple(reversed(labels))
    if isinstance(labels, list):
        return list(reversed(labels))
    return "[target-free audit replacement]"
