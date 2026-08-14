"""Contracts for deterministic, tensor-backed prepared data artifacts.

The normalizer in this module is deliberately fit-only-on-train.  It accepts
canonical samples, rather than arbitrary tensors, so a declared split is always
available for the leakage check.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from .schema import NeuralTextSample

PREPARED_ARTIFACT_VERSION = "1.0"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def tensor_checksum(values: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and values independent of device placement."""
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    cpu = values.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    # Keep this artifact contract dependency-light: the test/runtime package
    # does not require NumPy, so avoid Tensor.numpy() here.
    digest.update(bytes(cpu.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TensorBackedSample:
    """One canonical sample paired with its prepared ``[channels, time]`` tensor."""

    sample: NeuralTextSample
    values: torch.Tensor
    channel_mask: torch.Tensor | None = None
    time_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("prepared values must have shape [channels, time]")
        if self.values.shape[0] != self.sample.signal.channel_count:
            raise ValueError("prepared channel count must match sample.signal.channel_count")
        if self.values.shape[1] < 1:
            raise ValueError("prepared values need at least one time sample")
        if not self.values.is_floating_point():
            raise ValueError("prepared values must use a floating-point dtype")
        if not torch.isfinite(self.values).all():
            raise ValueError("prepared values must be finite")
        if self.channel_mask is not None:
            expected_channels = (self.values.shape[0],)
            invalid_channel_mask = (
                self.channel_mask.dtype != torch.bool
                or self.channel_mask.shape != expected_channels
            )
            if invalid_channel_mask:
                raise ValueError("channel_mask must be bool with shape [channels]")
            if not self.channel_mask.any():
                raise ValueError("at least one channel must be valid")
        if self.time_mask is not None:
            if self.time_mask.dtype != torch.bool or self.time_mask.shape != self.values.shape[1:]:
                raise ValueError("time_mask must be bool with shape [time]")
            if not self.time_mask.any():
                raise ValueError("at least one time sample must be valid")

    @property
    def resolved_channel_mask(self) -> torch.Tensor:
        if self.channel_mask is None:
            return torch.ones(self.values.shape[0], dtype=torch.bool, device=self.values.device)
        return self.channel_mask

    @property
    def resolved_time_mask(self) -> torch.Tensor:
        if self.time_mask is None:
            return torch.ones(self.values.shape[1], dtype=torch.bool, device=self.values.device)
        return self.time_mask

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.resolved_channel_mask.unsqueeze(-1) & self.resolved_time_mask.unsqueeze(0)

    @property
    def checksum(self) -> str:
        return _canonical_hash(
            {
                "values": tensor_checksum(self.values),
                "channel_mask": self.resolved_channel_mask.detach().cpu().tolist(),
                "time_mask": self.resolved_time_mask.detach().cpu().tolist(),
            }
        )


@dataclass(frozen=True, slots=True)
class ChannelNormalizer:
    """Per-channel affine statistics fit exclusively from declared train samples."""

    mean: torch.Tensor
    scale: torch.Tensor
    fit_sample_ids: tuple[str, ...]
    fit_split: str = "train"
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.ndim != 1:
            raise ValueError("normalizer mean and scale must be one-dimensional")
        if self.mean.shape != self.scale.shape or self.mean.numel() == 0:
            raise ValueError("normalizer mean and scale must be non-empty and same shape")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("normalizer statistics must be finite")
        if torch.any(self.scale <= 0):
            raise ValueError("normalizer scales must be positive")
        if self.fit_split != "train":
            raise ValueError("channel normalization may only be fit on the train split")
        if not self.fit_sample_ids or len(set(self.fit_sample_ids)) != len(self.fit_sample_ids):
            raise ValueError("normalizer requires unique train fit_sample_ids")
        if self.epsilon <= 0:
            raise ValueError("normalizer epsilon must be positive")

    @property
    def checksum(self) -> str:
        return _canonical_hash(
            {
                "mean": self.mean.detach().cpu().tolist(),
                "scale": self.scale.detach().cpu().tolist(),
                "fit_sample_ids": list(self.fit_sample_ids),
                "fit_split": self.fit_split,
                "epsilon": self.epsilon,
            }
        )

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Apply train-fitted affine normalization to a ``[channels, time]`` tensor."""
        if values.ndim != 2 or values.shape[0] != self.mean.numel():
            raise ValueError("values must be [normalizer channels, time]")
        mean = self.mean.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        scale = self.scale.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        return (values - mean) / scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_split": self.fit_split,
            "epsilon": self.epsilon,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChannelNormalizer:
        normalizer = cls(
            mean=torch.tensor(data["mean"], dtype=torch.float64),
            scale=torch.tensor(data["scale"], dtype=torch.float64),
            fit_sample_ids=tuple(str(value) for value in data["fit_sample_ids"]),
            fit_split=str(data.get("fit_split", "train")),
            epsilon=float(data.get("epsilon", 1e-6)),
        )
        expected = data.get("checksum")
        if expected is not None and expected != normalizer.checksum:
            raise ValueError("prepared normalizer checksum does not match its contents")
        return normalizer


def fit_train_channel_normalizer(
    samples: Iterable[TensorBackedSample], *, epsilon: float = 1e-6
) -> ChannelNormalizer:
    """Fit population mean/std from all and only canonical ``train`` samples."""
    rows = tuple(samples)
    if not rows:
        raise ValueError("cannot fit channel normalization without samples")
    non_train = [row.sample.sample_id for row in rows if row.sample.split != "train"]
    if non_train:
        raise ValueError(
            "channel normalization fit received non-train samples: " + ", ".join(sorted(non_train))
        )
    channels = rows[0].values.shape[0]
    if any(row.values.shape[0] != channels for row in rows):
        raise ValueError("all train tensors must have the same channel count")
    sums = torch.zeros(channels, dtype=torch.float64)
    squared_sums = torch.zeros(channels, dtype=torch.float64)
    counts = torch.zeros(channels, dtype=torch.float64)
    for row in rows:
        values = row.values.detach().to(dtype=torch.float64, device="cpu")
        valid = row.valid_mask.detach().to(device="cpu")
        masked_values = torch.where(valid, values, torch.zeros_like(values))
        sums += masked_values.sum(dim=1)
        squared_sums += masked_values.square().sum(dim=1)
        counts += valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each channel needs at least one valid train value")
    mean = sums / counts
    variance = (squared_sums / counts - mean.square()).clamp_min(0)
    scale = variance.sqrt().clamp_min(epsilon)
    return ChannelNormalizer(
        mean=mean,
        scale=scale,
        fit_sample_ids=tuple(sorted(row.sample.sample_id for row in rows)),
        epsilon=epsilon,
    )


@dataclass(frozen=True, slots=True)
class PreparedTensorRecord:
    """Serializable identity and checksum for a tensor-backed prepared sample."""

    sample_id: str
    split: str
    shape: tuple[int, int]
    dtype: str
    tensor_checksum: str

    @classmethod
    def from_sample(cls, row: TensorBackedSample) -> PreparedTensorRecord:
        if row.sample.split is None:
            raise ValueError("prepared artifact records require a declared sample split")
        return cls(
            sample_id=row.sample.sample_id,
            split=row.sample.split,
            shape=tuple(int(value) for value in row.values.shape),
            dtype=str(row.values.dtype),
            tensor_checksum=row.checksum,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tensor_checksum": self.tensor_checksum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreparedTensorRecord:
        shape = tuple(int(value) for value in data["shape"])
        if len(shape) != 2 or any(value < 1 for value in shape):
            raise ValueError("prepared record shape must be two positive dimensions")
        return cls(
            sample_id=str(data["sample_id"]),
            split=str(data["split"]),
            shape=shape,
            dtype=str(data["dtype"]),
            tensor_checksum=str(data["tensor_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class PreparedArtifactManifest:
    """Immutable summary of prepared tensors and the train-only fit that made them."""

    dataset_id: str
    source_manifest_checksum: str
    split_manifest_checksum: str
    records: tuple[PreparedTensorRecord, ...]
    normalizer: ChannelNormalizer
    version: str = PREPARED_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.version != PREPARED_ARTIFACT_VERSION:
            raise ValueError(f"unsupported prepared artifact version: {self.version!r}")
        if (
            not self.dataset_id
            or not self.source_manifest_checksum
            or not self.split_manifest_checksum
        ):
            raise ValueError(
                "prepared artifact manifest requires dataset and source/split checksums"
            )
        ids = [record.sample_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("prepared artifact manifest has duplicate sample IDs")

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": "openthought2text.prepared_tensor_artifact",
            "version": self.version,
            "dataset_id": self.dataset_id,
            "source_manifest_checksum": self.source_manifest_checksum,
            "split_manifest_checksum": self.split_manifest_checksum,
            "records": [record.to_dict() for record in self.records],
            "normalizer": self.normalizer.to_dict(),
        }
        if include_checksum:
            data["checksum"] = self.checksum
        return data

    @property
    def checksum(self) -> str:
        return _canonical_hash(self.to_dict(include_checksum=False))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreparedArtifactManifest:
        if data.get("kind") != "openthought2text.prepared_tensor_artifact":
            raise ValueError("not an OpenThought2Text prepared tensor artifact manifest")
        manifest = cls(
            dataset_id=str(data["dataset_id"]),
            source_manifest_checksum=str(data["source_manifest_checksum"]),
            split_manifest_checksum=str(data["split_manifest_checksum"]),
            records=tuple(PreparedTensorRecord.from_dict(row) for row in data["records"]),
            normalizer=ChannelNormalizer.from_dict(data["normalizer"]),
            version=str(data.get("version", PREPARED_ARTIFACT_VERSION)),
        )
        expected = data.get("checksum")
        if expected is not None and expected != manifest.checksum:
            raise ValueError("prepared artifact manifest checksum does not match its contents")
        return manifest


def build_prepared_artifact_manifest(
    samples: Iterable[TensorBackedSample], normalizer: ChannelNormalizer
) -> PreparedArtifactManifest:
    """Create a deterministic prepared-artifact manifest from tensor samples."""
    rows = tuple(samples)
    if not rows:
        raise ValueError("cannot build a prepared artifact manifest without samples")
    dataset_ids = {row.sample.dataset_id for row in rows}
    if len(dataset_ids) != 1:
        raise ValueError("prepared samples must have one dataset_id")
    train_ids = {row.sample.sample_id for row in rows if row.sample.split == "train"}
    if set(normalizer.fit_sample_ids) != train_ids:
        raise ValueError("normalizer fit sample IDs must exactly match artifact train sample IDs")
    if any(row.values.shape[0] != normalizer.mean.numel() for row in rows):
        raise ValueError("prepared tensor channels must match normalizer channels")
    records = tuple(
        sorted(
            (PreparedTensorRecord.from_sample(row) for row in rows),
            key=lambda row: row.sample_id,
        )
    )
    sorted_rows = sorted(rows, key=lambda row: row.sample.sample_id)
    source_rows = [row.sample.to_dict() for row in sorted_rows]
    split_rows = [
        {"sample_id": row.sample.sample_id, "split": row.sample.split} for row in sorted_rows
    ]
    return PreparedArtifactManifest(
        dataset_id=next(iter(dataset_ids)),
        source_manifest_checksum=_canonical_hash({"samples": source_rows}),
        split_manifest_checksum=_canonical_hash({"splits": split_rows}),
        records=records,
        normalizer=normalizer,
    )


def write_prepared_artifact_manifest(path: str | Path, manifest: PreparedArtifactManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_prepared_artifact_manifest(path: str | Path) -> PreparedArtifactManifest:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared artifact manifest: {source}") from error
    if not isinstance(data, dict):
        raise ValueError("prepared artifact manifest must be a JSON object")
    return PreparedArtifactManifest.from_dict(data)
