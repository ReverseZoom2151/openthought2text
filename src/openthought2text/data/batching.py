"""PyTorch dataset and collator for variable-sized prepared neural tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .prepared import TensorBackedSample


class VariableLengthTensorDataset(Dataset[TensorBackedSample]):
    """An immutable dataset that preserves canonical sample identity and masks."""

    def __init__(self, samples: Sequence[TensorBackedSample]) -> None:
        self._samples = tuple(samples)
        sample_ids = [row.sample.sample_id for row in self._samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("variable-length dataset requires unique sample IDs")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> TensorBackedSample:
        return self._samples[index]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample.sample_id for row in self._samples)


@dataclass(frozen=True, slots=True)
class NeuralTensorBatch:
    """Padded tensors plus explicit masks and source identities.

    ``signals`` is ``[batch, max_channels, max_time]``.  Values where either
    mask is false are always zeroed, so downstream models must use the masks
    rather than observing arbitrary padding contents.
    """

    signals: torch.Tensor
    channel_mask: torch.Tensor
    time_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    splits: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError("signals must have shape [batch, channels, time]")
        expected_channels = self.signals.shape[:2]
        expected_time = (self.signals.shape[0], self.signals.shape[2])
        if self.channel_mask.dtype != torch.bool or self.channel_mask.shape != expected_channels:
            raise ValueError("channel_mask must be bool with shape [batch, channels]")
        if self.time_mask.dtype != torch.bool or self.time_mask.shape != expected_time:
            raise ValueError("time_mask must be bool with shape [batch, time]")
        if (
            len(self.sample_ids) != self.signals.shape[0]
            or len(self.splits) != self.signals.shape[0]
        ):
            raise ValueError("sample_ids and splits must have one value per batch element")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("batch sample IDs must be unique")
        valid = self.channel_mask.unsqueeze(-1) & self.time_mask.unsqueeze(1)
        if torch.any(self.signals.masked_select(~valid) != 0):
            raise ValueError("signals outside channel/time masks must be zero")


def collate_tensor_backed_samples(rows: Sequence[TensorBackedSample]) -> NeuralTensorBatch:
    """Pad variable channel/time tensors and zero every masked input value."""
    if not rows:
        raise ValueError("cannot collate an empty batch")
    devices = {str(row.values.device) for row in rows}
    if len(devices) != 1:
        raise ValueError("all collated tensors must be on one device")
    dtype = rows[0].values.dtype
    if any(row.values.dtype != dtype for row in rows):
        raise ValueError("all collated tensors must have one dtype")
    max_channels = max(row.values.shape[0] for row in rows)
    max_time = max(row.values.shape[1] for row in rows)
    device = rows[0].values.device
    signals = torch.zeros((len(rows), max_channels, max_time), dtype=dtype, device=device)
    channel_mask = torch.zeros((len(rows), max_channels), dtype=torch.bool, device=device)
    time_mask = torch.zeros((len(rows), max_time), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        channels, time = row.values.shape
        current_channels = row.resolved_channel_mask.to(device=device)
        current_time = row.resolved_time_mask.to(device=device)
        valid = current_channels.unsqueeze(-1) & current_time.unsqueeze(0)
        clean = torch.where(valid, row.values, torch.zeros_like(row.values))
        signals[index, :channels, :time] = clean
        channel_mask[index, :channels] = current_channels
        time_mask[index, :time] = current_time
    return NeuralTensorBatch(
        signals=signals,
        channel_mask=channel_mask,
        time_mask=time_mask,
        sample_ids=tuple(row.sample.sample_id for row in rows),
        splits=tuple(row.sample.split for row in rows),
    )
