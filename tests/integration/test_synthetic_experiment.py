"""Regression coverage for the complete synthetic artifact trace."""

from __future__ import annotations

import ast
import runpy


def test_synthetic_experiment_runs_end_to_end(capsys) -> None:
    namespace = runpy.run_path("examples/synthetic_experiment.py")
    namespace["main"]()
    result = ast.literal_eval(capsys.readouterr().out.strip())
    assert result["train_loss"] > 0
    assert len(result["held_out_ids"]) == len(result["generated_token_ids"])
    assert len(result["checkpoint_sha256"]) == 64
