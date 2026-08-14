"""Target-free generated token IDs to auditable prediction-record conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openthought2text.controls import ControlCondition

from .audit import assert_target_free_signature
from .records import PredictionRecord


def generate_target_free_prediction_records(
    generator: Callable[[Any], Any],
    neural_input: Any,
    sample_ids: Sequence[str],
    decode_token_ids: Callable[[Sequence[int]], str],
    *,
    run_id: str,
    control: ControlCondition | str = ControlCondition.FULL,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[PredictionRecord, ...]:
    """Call ``generator(neural_input)`` and serialize its generated IDs.

    This API intentionally has no target/label argument.  Explicit target-like
    generator parameters are rejected before inference, and the generator is
    invoked exactly once with neural input as its sole positional argument.
    """
    assert_target_free_signature(generator)
    generated_ids = generator(neural_input)
    return token_ids_to_prediction_records(
        generated_ids,
        sample_ids,
        decode_token_ids,
        run_id=run_id,
        control=control,
        metadata=metadata,
    )


def token_ids_to_prediction_records(
    generated_ids: Any,
    sample_ids: Sequence[str],
    decode_token_ids: Callable[[Sequence[int]], str],
    *,
    run_id: str,
    control: ControlCondition | str = ControlCondition.FULL,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[PredictionRecord, ...]:
    """Decode a batch of generated token IDs into label-free ``PredictionRecord`` rows."""
    identifiers = tuple(sample_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("sample_ids must be non-empty and unique")
    rows = _token_rows(generated_ids, batch_size=len(identifiers))
    if len(rows) != len(identifiers):
        raise ValueError(f"generated {len(rows)} token rows for {len(identifiers)} sample IDs")
    control = ControlCondition(control)
    record_metadata = {"generation_path": "target_free_token_ids", **dict(metadata or {})}
    records: list[PredictionRecord] = []
    for sample_id, token_ids in zip(identifiers, rows, strict=True):
        decoded = decode_token_ids(token_ids)
        if not isinstance(decoded, str):
            raise TypeError("decode_token_ids must return text strings")
        records.append(
            PredictionRecord(
                sample_id=sample_id,
                prediction_text=decoded,
                run_id=run_id,
                control=control,
                target_free=True,
                metadata=record_metadata,
            )
        )
    return tuple(records)


def _token_rows(generated_ids: Any, *, batch_size: int) -> tuple[tuple[int, ...], ...]:
    value = _to_list(generated_ids)
    if not isinstance(value, (list, tuple)):
        raise TypeError("generated IDs must be a one- or two-dimensional token sequence")
    if batch_size == 1 and (not value or isinstance(value[0], int)):
        return (_token_row(value),)
    if len(value) != batch_size or any(not isinstance(row, (list, tuple)) for row in value):
        raise ValueError("generated token IDs must have shape [batch, tokens]")
    return tuple(_token_row(row) for row in value)


def _token_row(row: Sequence[Any]) -> tuple[int, ...]:
    parsed: list[int] = []
    for token in row:
        if isinstance(token, bool):
            raise TypeError("token IDs must be integers, not booleans")
        numeric = int(token)
        if numeric != token:
            raise ValueError("token IDs must be integer-valued")
        parsed.append(numeric)
    return tuple(parsed)


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
