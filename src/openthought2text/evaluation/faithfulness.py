"""End-to-end evaluation of neural grounding against declared controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openthought2text.controls import ControlCondition, build_control

from .audit import LabelInvarianceResult, audit_label_invariance, forbidden_generation_parameters
from .grounding import GroundingReport, build_grounding_report


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A scalar metric with a declared direction for grounded-gain calculation."""

    name: str
    evaluate: Callable[[Sequence[str], Sequence[str]], float]
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name must be non-empty")


@dataclass(frozen=True, slots=True)
class FaithfulnessConditionResult:
    condition: ControlCondition
    predictions: tuple[str, ...]
    scores: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class GenerationAuditSummary:
    forbidden_parameters: tuple[str, ...]
    label_invariance: LabelInvarianceResult | None

    @property
    def signature_is_target_free(self) -> bool:
        return not self.forbidden_parameters

    @property
    def passed(self) -> bool:
        return self.signature_is_target_free and (
            self.label_invariance is None or self.label_invariance.invariant
        )


@dataclass(frozen=True, slots=True)
class FaithfulnessSuiteResult:
    conditions: tuple[FaithfulnessConditionResult, ...]
    grounding: Mapping[str, GroundingReport]
    audit: GenerationAuditSummary

    def condition(self, name: ControlCondition | str) -> FaithfulnessConditionResult:
        parsed = ControlCondition(name)
        for result in self.conditions:
            if result.condition is parsed:
                return result
        raise KeyError(f"control was not evaluated: {parsed.value}")


DEFAULT_SIGNAL_CONTROLS = (
    ControlCondition.FULL,
    ControlCondition.SHUFFLED,
    ControlCondition.ZERO,
    ControlCondition.GAUSSIAN_NOISE,
    ControlCondition.MASK_ONLY,
    ControlCondition.LENGTH_ONLY,
    ControlCondition.TIMING_ONLY,
    ControlCondition.PHASE_SURROGATE,
)


def run_faithfulness_suite(
    generator: Callable[[Any], Sequence[str] | str],
    signal: Any,
    references: Sequence[str],
    metrics: Sequence[MetricSpec],
    *,
    controls: Sequence[ControlCondition | str] = DEFAULT_SIGNAL_CONTROLS,
    control_context: Mapping[str, Any] | None = None,
    seed: int = 0,
    audit_labels: Any | None = None,
    audit_target_keyword: str = "labels",
) -> FaithfulnessSuiteResult:
    """Evaluate target-free predictions under full and control signal inputs.

    ``control_context`` is passed only to named control builders and must state any
    retained side information, e.g. ``valid_mask``, ``valid_lengths``,
    ``event_indices``, ``channels``, and ``time_steps``.  A requested structural
    control fails loudly if that declaration is incomplete.
    """
    if not metrics:
        raise ValueError("at least one metric is required")
    metric_names = [metric.name for metric in metrics]
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("metric names must be unique")
    parsed_controls = tuple(ControlCondition(item) for item in controls)
    if ControlCondition.FULL not in parsed_controls:
        raise ValueError("faithfulness suite must include the full-signal condition")
    if ControlCondition.SHUFFLED not in parsed_controls:
        raise ValueError("faithfulness suite must include the shuffled-signal condition")
    if len(set(parsed_controls)) != len(parsed_controls):
        raise ValueError("each control condition may be evaluated only once")
    context = dict(control_context or {})
    results: list[FaithfulnessConditionResult] = []
    for offset, condition in enumerate(parsed_controls):
        control_signal = build_control(condition, signal, seed=seed + offset, **context)
        predictions = _predictions(generator(control_signal))
        if len(predictions) != len(references):
            raise ValueError(
                f"{condition.value} returned {len(predictions)} predictions for {len(references)} references"
            )
        scores = {
            metric.name: float(metric.evaluate(predictions, references)) for metric in metrics
        }
        results.append(FaithfulnessConditionResult(condition, predictions, scores))

    full = next(result for result in results if result.condition is ControlCondition.FULL)
    controls_by_name = {result.condition.value: result for result in results if result is not full}
    grounding = {
        metric.name: build_grounding_report(
            full.scores[metric.name],
            {name: result.scores[metric.name] for name, result in controls_by_name.items()},
            shuffled_score=controls_by_name[ControlCondition.SHUFFLED.value].scores[metric.name],
            higher_is_better=metric.higher_is_better,
        )
        for metric in metrics
    }
    labels = references if audit_labels is None else audit_labels
    audit = GenerationAuditSummary(
        forbidden_parameters=forbidden_generation_parameters(generator),
        label_invariance=audit_label_invariance(
            generator, signal, labels, target_keyword=audit_target_keyword
        ),
    )
    return FaithfulnessSuiteResult(tuple(results), grounding, audit)


def _predictions(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
