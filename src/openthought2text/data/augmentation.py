"""Deterministic, alignment-aware neural-only augmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .batching import NeuralTensorBatch
from .prepared import TensorBackedSample


@dataclass(frozen=True, slots=True)
class NeuralAugmentationConfig:
    """Augmentations that never modify labels, IDs, intervals, or split metadata.

    Masks and noise preserve the original time index.  A nonzero time shift is
    deliberately opt-in and only permitted for samples declaring
    ``metadata['time_shift_alignment_safe'] == True``; otherwise it would alter
    neural/label timing and is rejected.
    """

    temporal_mask_probability: float = 0.0
    channel_dropout_probability: float = 0.0
    additive_noise_std: float = 0.0
    max_time_shift: int = 0

    def __post_init__(self) -> None:
        for name in ("temporal_mask_probability", "channel_dropout_probability"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.additive_noise_std < 0:
            raise ValueError("additive_noise_std must be non-negative")
        if self.max_time_shift < 0:
            raise ValueError("max_time_shift must be non-negative")


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _augment_values(
    values: torch.Tensor,
    channel_mask: torch.Tensor,
    time_mask: torch.Tensor,
    config: NeuralAugmentationConfig,
    generator: torch.Generator,
    *,
    time_shift_safe: bool,
) -> torch.Tensor:
    valid = channel_mask.unsqueeze(-1) & time_mask.unsqueeze(0)
    result = torch.where(valid, values, torch.zeros_like(values))
    if config.temporal_mask_probability:
        selected_times = (torch.rand(time_mask.shape, device=values.device, generator=generator)
                          < config.temporal_mask_probability) & time_mask
        result[:, selected_times] = 0
    if config.channel_dropout_probability:
        dropped = (torch.rand(channel_mask.shape, device=values.device, generator=generator)
                   < config.channel_dropout_probability) & channel_mask
        # Keep at least one observed channel whenever one was available.
        if dropped[channel_mask].all():
            available = torch.nonzero(channel_mask, as_tuple=False).flatten()
            keep = available[torch.randint(len(available), (1,), generator=generator, device=values.device)]
            dropped[keep] = False
        result[dropped, :] = 0
    if config.additive_noise_std:
        noise = torch.randn(result.shape, dtype=result.dtype, device=result.device, generator=generator)
        result = torch.where(valid, result + noise * config.additive_noise_std, torch.zeros_like(result))
    if config.max_time_shift:
        if not time_shift_safe:
            raise ValueError("nonzero time shift requires declared time_shift_alignment_safe metadata")
        shift = int(torch.randint(-config.max_time_shift, config.max_time_shift + 1, (1,), generator=generator, device=values.device))
        # Roll only the observed temporal region; masked/padded values stay zero.
        indices = torch.nonzero(time_mask, as_tuple=False).flatten()
        result[:, indices] = result[:, indices].roll(shift, dims=1)
    return torch.where(valid, result, torch.zeros_like(result))


def augment_tensor_backed_samples(
    samples: Sequence[TensorBackedSample], config: NeuralAugmentationConfig, *, seed: int
) -> tuple[TensorBackedSample, ...]:
    """Apply seeded neural-only augmentation while retaining each exact sample label."""
    result: list[TensorBackedSample] = []
    for index, row in enumerate(samples):
        generator = _generator(seed + index, row.values.device)
        values = _augment_values(
            row.values,
            row.resolved_channel_mask,
            row.resolved_time_mask,
            config,
            generator,
            time_shift_safe=row.sample.metadata.get("time_shift_alignment_safe") is True,
        )
        result.append(replace(row, values=values))
    return tuple(result)


def augment_neural_tensor_batch(
    batch: NeuralTensorBatch, config: NeuralAugmentationConfig, *, seed: int
) -> NeuralTensorBatch:
    """Augment a padded batch without ever exposing or altering padded values.

    Batches do not carry per-sample alignment declarations, so nonzero shifts
    are rejected here even if they were allowed during individual preparation.
    """
    if config.max_time_shift:
        raise ValueError("time shifts require per-sample alignment metadata; augment samples before collation")
    signals = torch.zeros_like(batch.signals)
    for index in range(batch.signals.shape[0]):
        generator = _generator(seed + index, batch.signals.device)
        signals[index] = _augment_values(
            batch.signals[index], batch.channel_mask[index], batch.time_mask[index], config, generator,
            time_shift_safe=False,
        )
    return NeuralTensorBatch(
        signals=signals,
        channel_mask=batch.channel_mask,
        time_mask=batch.time_mask,
        sample_ids=batch.sample_ids,
        splits=batch.splits,
    )
