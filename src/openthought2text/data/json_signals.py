"""Safe JSON-only loading of portable signal matrices into prepared samples."""

from __future__ import annotations

from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

import torch

from .manifest import DatasetManifest
from .prepared import TensorBackedSample
from .schema import NeuralTextSample


_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def select_split_samples(
    samples: Iterable[NeuralTextSample], split: ManifestSplit | str
) -> tuple[NeuralTextSample, ...]:
    """Select canonical train/validation/test samples in source order.

    ``val`` is accepted only as an input convenience and resolves to the
    canonical manifest label ``validation``.
    """
    value = "validation" if split == "val" else split
    try:
        resolved = ManifestSplit(value)
    except ValueError as error:
        raise ValueError("split must be train, validation (or val), or test") from error
    return tuple(sample for sample in samples if sample.split == resolved.value)


def _artifact_root(root: str | Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("signal artifact root does not exist or is not a directory")
    return resolved


def _resolve_json_reference(root: Path, sample: NeuralTextSample) -> Path:
    uri = sample.signal.uri
    if "://" in uri or uri.startswith("file:"):
        raise ValueError(f"sample {sample.sample_id} signal URI must be a local relative path")
    candidate = Path(uri)
    if candidate.is_absolute():
        raise ValueError(f"sample {sample.sample_id} signal URI must be relative to artifact root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"sample {sample.sample_id} signal URI escapes artifact root") from error
    if resolved.suffix.casefold() != ".json":
        raise ValueError(f"sample {sample.sample_id} signal URI must reference a .json file")
    if not resolved.is_file():
        raise ValueError(f"sample {sample.sample_id} signal file is missing: {resolved}")
    return resolved


def _verify_checksum(path: Path, sample: NeuralTextSample) -> None:
    expected = sample.signal.checksum_sha256
    if expected is None:
        return
    if _CHECKSUM_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"sample {sample.sample_id} signal checksum is not SHA-256 hex")
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"sample {sample.sample_id} signal checksum does not match file bytes")


def _json_matrix(path: Path, sample: NeuralTextSample) -> torch.Tensor:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"sample {sample.sample_id} signal file is invalid JSON") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"sample {sample.sample_id} JSON signal must be a non-empty channel list")
    if len(payload) != sample.signal.channel_count:
        raise ValueError(
            f"sample {sample.sample_id} expected {sample.signal.channel_count} channels, "
            f"found {len(payload)}"
        )
    width: int | None = None
    matrix: list[list[float]] = []
    for channel_index, channel in enumerate(payload):
        if not isinstance(channel, list) or not channel:
            raise ValueError(
                f"sample {sample.sample_id} channel {channel_index} is not a non-empty list"
            )
        if width is None:
            width = len(channel)
        elif len(channel) != width:
            raise ValueError(
                f"sample {sample.sample_id} JSON channels have inconsistent time lengths"
            )
        row: list[float] = []
        for value in channel:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"sample {sample.sample_id} JSON signal contains a non-numeric value"
                )
            numeric = float(value)
            if not torch.isfinite(torch.tensor(numeric)):
                raise ValueError(
                    f"sample {sample.sample_id} JSON signal contains a non-finite value"
                )
            row.append(numeric)
        matrix.append(row)
    return torch.tensor(matrix, dtype=torch.float32)


def load_json_tensor_samples(
    manifest: DatasetManifest,
    root: str | Path,
    *,
    split: ManifestSplit | str | None = None,
) -> tuple[TensorBackedSample, ...]:
    """Load direct JSON ``[channels, time]`` matrices from a portable manifest.

    The only accepted tensor representation is JSON; ``.pt``, pickle, URLs,
    absolute paths, and paths outside ``root`` are rejected before any payload
    decoding takes place.
    """
    artifact_root = _artifact_root(root)
    rows = manifest.samples if split is None else select_split_samples(manifest.samples, split)
    loaded: list[TensorBackedSample] = []
    for sample in rows:
        path = _resolve_json_reference(artifact_root, sample)
        _verify_checksum(path, sample)
        loaded.append(TensorBackedSample(sample=sample, values=_json_matrix(path, sample)))
    return tuple(loaded)
