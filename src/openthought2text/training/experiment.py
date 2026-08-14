"""Reusable, leakage-aware mechanics for a supervised neural-to-text run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from openthought2text.data import (
    ChannelNormalizer,
    NeuralTensorBatch,
    TensorBackedSample,
    TrainTextTokenizer,
    collate_tensor_backed_samples,
    fit_train_channel_normalizer,
    fit_train_text_tokenizer,
)
from openthought2text.models import NeuralToTextModel

from .supervised import SupervisedStepResult, supervised_train_step


@dataclass(frozen=True, slots=True)
class TrainingInputs:
    """Train-only learned artifacts and normalized training examples."""

    tokenizer: TrainTextTokenizer
    normalizer: ChannelNormalizer
    rows: tuple[TensorBackedSample, ...]


def build_training_inputs(
    rows: Iterable[TensorBackedSample], *, unknown_policy: str = "unk"
) -> TrainingInputs:
    """Fit every learned preprocessing/text artifact only on declared train rows."""
    source = tuple(rows)
    if not source:
        raise ValueError("training inputs require at least one row")
    if any(row.sample.split != "train" for row in source):
        raise ValueError("training inputs may contain only samples declared train")
    normalizer = fit_train_channel_normalizer(source)
    tokenizer = fit_train_text_tokenizer(
        (row.sample for row in source), unknown_policy=unknown_policy
    )
    normalized = tuple(
        TensorBackedSample(
            row.sample,
            normalizer.apply(row.values).to(dtype=row.values.dtype),
            row.channel_mask,
            row.time_mask,
        )
        for row in source
    )
    return TrainingInputs(tokenizer=tokenizer, normalizer=normalizer, rows=normalized)


def target_ids_for_batch(
    batch: NeuralTensorBatch,
    samples: Mapping[str, TensorBackedSample],
    tokenizer: TrainTextTokenizer,
) -> torch.Tensor:
    """Construct padded teacher-forcing IDs strictly from named training rows."""
    sequences: list[tuple[int, ...]] = []
    for sample_id in batch.sample_ids:
        row = samples.get(sample_id)
        if row is None:
            raise ValueError(f"batch sample has no supplied source row: {sample_id}")
        if row.sample.split != "train" or row.sample.target is None:
            raise ValueError(f"teacher-forcing target is not authorized for sample: {sample_id}")
        sequences.append(tokenizer.encode(row.sample.target.text))
    result = torch.full(
        (len(sequences), max(map(len, sequences))),
        -100,
        dtype=torch.long,
        device=batch.signals.device,
    )
    for index, sequence in enumerate(sequences):
        result[index, : len(sequence)] = torch.tensor(sequence, device=result.device)
    return result


def train_one_epoch(
    model: NeuralToTextModel,
    rows: Sequence[TensorBackedSample],
    tokenizer: TrainTextTokenizer,
    *,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    sample_rate_hz: float = 200.0,
) -> tuple[SupervisedStepResult, ...]:
    """Run one ordered epoch; no validation/test text is accepted by this API."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not rows or any(row.sample.split != "train" for row in rows):
        raise ValueError("epoch rows must be a non-empty all-train collection")
    by_id = {row.sample.sample_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("epoch rows require unique sample IDs")
    results = []
    for start in range(0, len(rows), batch_size):
        batch = collate_tensor_backed_samples(rows[start : start + batch_size])
        results.append(
            supervised_train_step(
                model,
                batch,
                target_ids_for_batch(batch, by_id, tokenizer),
                optimizer=optimizer,
                sample_rate_hz=sample_rate_hz,
            )
        )
    return tuple(results)
