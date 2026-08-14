from __future__ import annotations

import torch

from openthought2text.cli.main import main
from openthought2text.config.run import RunManifest
from openthought2text.losses.composite import compose_losses


def test_cli_accepts_data_discovery(tmp_path) -> None:
    assert main(["data", "discover", "--dataset", "synthetic", "--root", str(tmp_path)]) == 0


def test_run_manifest_is_serializable() -> None:
    manifest = RunManifest(
        experiment_name="tiny",
        dataset_artifact_checksum="dataset",
        split_manifest_checksum="split",
        seed=7,
        resolved_config={"model": "tiny"},
    )
    assert manifest.to_dict()["seed"] == 7


def test_composite_losses_requires_explicit_weights() -> None:
    losses = {"sequence": torch.tensor(2.0), "contrast": torch.tensor(3.0)}
    assert compose_losses(losses, {"sequence": 0.5, "contrast": 2.0}).item() == 7.0
