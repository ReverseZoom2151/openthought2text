"""Integration-grade evaluation from saved prediction artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openthought2text.controls import ControlCondition

from .grounding import build_grounding_report
from .metrics import corpus_character_error_rate, corpus_word_error_rate, retrieval_metrics
from .records import (
    BenchmarkRowLabel,
    ControlResult,
    EvaluationReport,
    PredictionRecord,
    read_prediction_jsonl,
)


@dataclass(frozen=True, slots=True)
class RetrievalInputs:
    """Scores and correct-candidate indices for one evaluated control condition."""

    score_rows: Sequence[Sequence[float]]
    positive_indices: Sequence[int]
    ks: Sequence[int] = (1, 5, 10)


def evaluate_saved_predictions(
    predictions: Iterable[PredictionRecord] | str | Path,
    *,
    benchmark: BenchmarkRowLabel,
    references: Mapping[str, str] | None = None,
    retrieval_by_control: Mapping[ControlCondition | str, RetrievalInputs] | None = None,
    prediction_artifact: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Build a report from saved JSONL rows or in-memory prediction records.

    Prediction records must be matched by sample ID across every included signal
    control.  This makes a full-versus-control gain a paired comparison rather
    than an accidental comparison of different held-out examples.
    """
    records, inferred_artifact = _load_records(predictions)
    if not records:
        raise ValueError("prediction evaluation requires at least one record")
    run_ids = {record.run_id for record in records}
    if len(run_ids) != 1:
        raise ValueError("all prediction records in one evaluation report must share run_id")
    grouped = _group_records(records)
    if ControlCondition.FULL not in grouped:
        raise ValueError("evaluation requires full-signal prediction records")
    _validate_paired_sample_ids(grouped)
    resolved_references = _resolve_references(grouped, references)
    retrieval_inputs = {
        ControlCondition(condition): value
        for condition, value in (retrieval_by_control or {}).items()
    }

    control_results: list[ControlResult] = []
    for condition in sorted(grouped, key=lambda item: item.value):
        rows = grouped[condition]
        ordered_ids = sorted(rows)
        predictions_for_condition = [rows[sample_id].prediction_text for sample_id in ordered_ids]
        references_for_condition = [resolved_references[sample_id] for sample_id in ordered_ids]
        scores = {
            "cer": corpus_character_error_rate(
                references_for_condition, predictions_for_condition
            ).rate,
            "wer": corpus_word_error_rate(references_for_condition, predictions_for_condition).rate,
        }
        if condition in retrieval_inputs:
            if len(retrieval_inputs[condition].score_rows) != len(rows):
                raise ValueError(
                    f"{condition.value} retrieval scores must have one row per prediction record"
                )
            retrieval = retrieval_metrics(
                retrieval_inputs[condition].score_rows,
                retrieval_inputs[condition].positive_indices,
                ks=retrieval_inputs[condition].ks,
            )
            scores["retrieval_mrr"] = retrieval.mean_reciprocal_rank
            scores["retrieval_mean_rank"] = retrieval.mean_rank
            scores.update(
                {f"retrieval_recall_at_{k}": value for k, value in retrieval.recall_at.items()}
            )
        control_results.append(ControlResult(condition, scores, examples=len(rows)))

    full = next(row for row in control_results if row.condition is ControlCondition.FULL)
    results_by_condition = {row.condition: row for row in control_results}
    grounding = {}
    if ControlCondition.SHUFFLED in results_by_condition:
        for metric, full_score in full.scores.items():
            paired_controls = {
                condition.value: result.scores[metric]
                for condition, result in results_by_condition.items()
                if condition is not ControlCondition.FULL and metric in result.scores
            }
            if paired_controls and ControlCondition.SHUFFLED.value in paired_controls:
                grounding[metric] = build_grounding_report(
                    full_score,
                    paired_controls,
                    shuffled_score=paired_controls[ControlCondition.SHUFFLED.value],
                    higher_is_better=not metric.endswith("cer")
                    and not metric.endswith("wer")
                    and metric != "retrieval_mean_rank",
                )
    return EvaluationReport(
        run_id=next(iter(run_ids)),
        benchmark=benchmark,
        metrics=full.scores,
        prediction_count=full.examples,
        prediction_artifact=prediction_artifact or inferred_artifact,
        control_results=tuple(control_results),
        grounding=grounding,
        metadata=dict(metadata or {}),
    )


def _load_records(
    predictions: Iterable[PredictionRecord] | str | Path,
) -> tuple[tuple[PredictionRecord, ...], str]:
    if isinstance(predictions, (str, Path)):
        path = Path(predictions)
        return read_prediction_jsonl(path), str(path)
    return tuple(predictions), "in_memory://prediction_records"


def _group_records(
    records: Sequence[PredictionRecord],
) -> dict[ControlCondition, dict[str, PredictionRecord]]:
    grouped: dict[ControlCondition, dict[str, PredictionRecord]] = defaultdict(dict)
    for record in records:
        if record.sample_id in grouped[record.control]:
            raise ValueError(
                f"duplicate sample_id within {record.control.value}: {record.sample_id}"
            )
        grouped[record.control][record.sample_id] = record
    return dict(grouped)


def _validate_paired_sample_ids(
    grouped: Mapping[ControlCondition, Mapping[str, PredictionRecord]],
) -> None:
    full_ids = set(grouped[ControlCondition.FULL])
    for condition, records in grouped.items():
        if set(records) != full_ids:
            raise ValueError(
                f"{condition.value} does not contain exactly the full-signal sample IDs; "
                "control comparisons must be paired"
            )


def _resolve_references(
    grouped: Mapping[ControlCondition, Mapping[str, PredictionRecord]],
    external: Mapping[str, str] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for sample_id, full_record in grouped[ControlCondition.FULL].items():
        inline = full_record.reference_text
        supplied = None if external is None else external.get(sample_id)
        if inline is not None and supplied is not None and inline != supplied:
            raise ValueError(
                f"conflicting inline and supplied reference for sample_id: {sample_id}"
            )
        reference = inline if inline is not None else supplied
        if reference is None:
            raise ValueError(f"missing reference for sample_id: {sample_id}")
        for condition, records in grouped.items():
            control_reference = records[sample_id].reference_text
            if control_reference is not None and control_reference != reference:
                raise ValueError(
                    f"conflicting reference in {condition.value} control for sample_id: {sample_id}"
                )
        result[sample_id] = reference
    return result
