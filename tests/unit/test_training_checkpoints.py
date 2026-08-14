from __future__ import annotations

import torch

from openthought2text.config.run import RunManifest
from openthought2text.training import CheckpointMetadata, load_checkpoint_metadata, save_checkpoint


def test_checkpoint_retains_run_manifest(tmp_path) -> None:
    model = torch.nn.Linear(2, 3)
    metadata = CheckpointMetadata(
        epoch=2,
        step=11,
        selection_metric="grounded_gain",
        selection_value=0.42,
        run_manifest=RunManifest("tiny", "dataset", "split", 7, {"model": "tiny"}),
    )
    path = save_checkpoint(tmp_path / "checkpoint.pt", model=model, metadata=metadata)
    loaded = load_checkpoint_metadata(path)
    assert loaded["selection_metric"] == "grounded_gain"
    assert loaded["run_manifest"]["seed"] == 7
