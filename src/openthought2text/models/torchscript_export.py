"""Safe, scoped TorchScript export for target-free neural evidence encoding.

Autoregressive ``NeuralToTextModel.generate`` has dynamic stopping behavior and
is intentionally outside this initial export scope.  This module exports the
deterministic encoder-evidence path only and never loads/deserializes artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import checkpoint_architecture_metadata
from .model import NeuralToTextModel


TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE = "neural_encoder_evidence_v1"


class NeuralEncoderEvidenceTorchScriptAdapter(nn.Module):
    """Tensor-only, target-free adapter around ``NeuralToTextModel.encoder``."""

    def __init__(self, model: NeuralToTextModel) -> None:
        super().__init__()
        self.encoder = model.encoder

    def forward(
        self,
        signals: torch.Tensor,
        sample_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(
            signals,
            sample_mask=sample_mask,
            channel_mask=channel_mask,
            coordinates=coordinates,
            sample_rate_hz=1.0,
        )
        return encoded.features, encoded.mask, encoded.timing.start, encoded.timing.end


@dataclass(frozen=True)
class TorchScriptExportValidation:
    """Validation result and in-memory artifact; callers choose whether to save."""

    scope: str
    exportable: bool
    errors: tuple[str, ...]
    architecture_metadata: dict[str, Any] | None
    input_signature: dict[str, dict[str, Any]] | None
    scripted_module: torch.jit.ScriptModule | torch.jit.TopLevelTracedModule | None

    def raise_if_invalid(self) -> None:
        if not self.exportable:
            raise ValueError("TorchScript export validation failed: " + "; ".join(self.errors))


def _compare_outputs(
    eager: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    traced: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for eager_value, traced_value in zip(eager, traced):
        if eager_value.dtype == torch.bool or eager_value.dtype in (torch.int32, torch.int64):
            if not torch.equal(eager_value, traced_value):
                raise ValueError("traced output differs from eager output")
        else:
            torch.testing.assert_close(eager_value, traced_value, rtol=1e-4, atol=1e-5)


def _input_signature(
    signals: torch.Tensor,
    sample_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    coordinates: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    """Record the traced tensor contract; this first export is shape-scoped."""
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in {
            "signals": signals,
            "sample_mask": sample_mask,
            "channel_mask": channel_mask,
            "coordinates": coordinates,
        }.items()
    }


def validate_neural_encoder_evidence_torchscript(
    model: NeuralToTextModel,
    signals: torch.Tensor,
    sample_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    coordinates: torch.Tensor,
) -> TorchScriptExportValidation:
    """Trace and compare the fixed-input encoder evidence scope without saving.

    ``model`` must originate from the config factory so the returned sidecar
    metadata has a state-dict-compatible architecture fingerprint.
    """
    try:
        metadata = checkpoint_architecture_metadata(model)
    except (TypeError, ValueError, RuntimeError) as error:
        return TorchScriptExportValidation(
            scope=TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE,
            exportable=False,
            errors=(f"architecture metadata unavailable: {error}",),
            architecture_metadata=None,
            input_signature=None,
            scripted_module=None,
        )
    signature = _input_signature(signals, sample_mask, channel_mask, coordinates)
    adapter = NeuralEncoderEvidenceTorchScriptAdapter(model)
    was_training = model.training
    try:
        model.eval()
        inputs = (signals, sample_mask, channel_mask, coordinates)
        with torch.no_grad():
            eager = adapter(*inputs)
            traced = torch.jit.trace(adapter, inputs, strict=True, check_trace=True)
            traced_output = traced(*inputs)
            _compare_outputs(eager, traced_output)
    except (RuntimeError, TypeError, ValueError) as error:
        return TorchScriptExportValidation(
            scope=TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE,
            exportable=False,
            errors=(str(error),),
            architecture_metadata=metadata,
            input_signature=signature,
            scripted_module=None,
        )
    finally:
        model.train(was_training)
    return TorchScriptExportValidation(
        scope=TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE,
        exportable=True,
        errors=(),
        architecture_metadata=metadata,
        input_signature=signature,
        scripted_module=traced,
    )


def export_neural_encoder_evidence_torchscript(
    model: NeuralToTextModel,
    signals: torch.Tensor,
    sample_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    coordinates: torch.Tensor,
    artifact_path: str | Path,
) -> TorchScriptExportValidation:
    """Validate then save a TorchScript encoder and JSON architecture sidecar.

    The sidecar is written at ``<artifact_path>.metadata.json``.  This function
    does not open or load any existing checkpoint/artifact.
    """
    validation = validate_neural_encoder_evidence_torchscript(
        model, signals, sample_mask, channel_mask, coordinates
    )
    validation.raise_if_invalid()
    assert validation.scripted_module is not None and validation.architecture_metadata is not None
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(validation.scripted_module, str(path))
    sidecar = {
        "torchscript_scope": validation.scope,
        "architecture_metadata": validation.architecture_metadata,
        "input_signature": validation.input_signature,
    }
    sidecar_path = Path(f"{path}.metadata.json")
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return validation
