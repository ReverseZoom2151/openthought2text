"""Run metadata required for reproducible experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    """Immutable summary written beside checkpoints and predictions."""

    experiment_name: str
    dataset_artifact_checksum: str
    split_manifest_checksum: str
    seed: int
    resolved_config: dict[str, Any]
    code_revision: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
