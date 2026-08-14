"""Safe-by-default checkpoint metadata and local state persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from openthought2text.config.run import RunManifest


@dataclass(frozen=True)
class CheckpointMetadata:
    """Selection and compatibility information stored beside model state."""

    epoch: int
    step: int
    selection_metric: str
    selection_value: float
    run_manifest: RunManifest

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.step < 0:
            raise ValueError("epoch and step must be non-negative")
        if not self.selection_metric.strip():
            raise ValueError("selection_metric must be non-empty")


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Atomically save local trusted training state and reproducibility metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "openthought2text.checkpoint.v1",
        "model_state": model.state_dict(),
        "metadata": asdict(metadata),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read metadata from a **trusted local** checkpoint without restoring a model."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "openthought2text.checkpoint.v1":
        raise ValueError("unsupported checkpoint format")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing metadata")
    return metadata
