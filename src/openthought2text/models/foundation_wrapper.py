"""Contracted wrapper for already-instantiated external neural encoders.

No loader, path, URL, checkpoint format, or deserialization utility belongs in
this module.  Callers provide an already-created ``nn.Module`` and explicit
pretraining provenance before it can participate in the model graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .types import NeuralEncoderOutput, TokenTiming

OverlapLabel = Literal["disjoint", "unknown", "potential_overlap", "same_dataset"]


@dataclass(frozen=True)
class FoundationFeatureContract:
    """Explicit feature/mask/timing boundary for an imported representation."""

    input_feature_size: int
    output_feature_size: int
    feature_layout: str = "batch_tokens_features"
    mask_semantics: str = "true_is_valid"
    timing_policy: str = "passthrough"

    def __post_init__(self) -> None:
        if self.input_feature_size < 1 or self.output_feature_size < 1:
            raise ValueError("foundation feature sizes must be positive")
        if self.feature_layout != "batch_tokens_features":
            raise ValueError("feature_layout must be batch_tokens_features")
        if self.mask_semantics != "true_is_valid":
            raise ValueError("mask_semantics must be true_is_valid")
        if self.timing_policy != "passthrough":
            raise ValueError("timing_policy must be passthrough")


@dataclass(frozen=True)
class FoundationPretrainingProvenance:
    """Declared source and overlap status; unknown is explicit rather than silent."""

    source_name: str
    overlap_label: OverlapLabel
    pretraining_description: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ValueError("source_name must be nonempty")
        if self.overlap_label not in {"disjoint", "unknown", "potential_overlap", "same_dataset"}:
            raise ValueError(
                "overlap_label must be disjoint, unknown, potential_overlap, or same_dataset"
            )
        if (
            not isinstance(self.pretraining_description, str)
            or not self.pretraining_description.strip()
        ):
            raise ValueError("pretraining_description must be nonempty")


class FoundationEncoderWrapper(nn.Module):
    """Apply an external feature module while enforcing a declared contract.

    ``external_encoder`` must implement ``forward(features, mask)`` and return
    ``[batch, tokens, contract.output_feature_size]``.  Inputs are zeroed at
    invalid positions before that call and output is zeroed afterward, making
    padded source values inert.  Frozen modules retain gradients to incoming
    feature tensors but cannot accumulate parameter gradients.
    """

    def __init__(
        self,
        external_encoder: nn.Module,
        feature_contract: FoundationFeatureContract,
        provenance: FoundationPretrainingProvenance,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(external_encoder, nn.Module):
            raise ValueError("external_encoder must be an already-instantiated nn.Module")
        if not isinstance(feature_contract, FoundationFeatureContract):
            raise ValueError("feature_contract must be FoundationFeatureContract")
        if not isinstance(provenance, FoundationPretrainingProvenance):
            raise ValueError("provenance must be FoundationPretrainingProvenance")
        self.external_encoder = external_encoder
        self.feature_contract = feature_contract
        self.provenance = provenance
        self._trainable = False
        self.set_trainable(trainable)

    @property
    def trainable(self) -> bool:
        return self._trainable

    def set_trainable(self, trainable: bool) -> None:
        if not isinstance(trainable, bool):
            raise ValueError("trainable must be a boolean")
        self._trainable = trainable
        for parameter in self.external_encoder.parameters():
            parameter.requires_grad_(trainable)
        if not trainable:
            self.external_encoder.eval()

    def train(self, mode: bool = True) -> FoundationEncoderWrapper:
        super().train(mode)
        if not self._trainable:
            self.external_encoder.eval()
        return self

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        timing: TokenTiming,
    ) -> NeuralEncoderOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_contract.input_feature_size:
            raise ValueError("features must be [batch, tokens, contract.input_feature_size]")
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must be [batch, tokens]")
        if timing.start.shape != mask.shape:
            raise ValueError("timing must match feature batch/token axes")
        valid = mask.bool()
        masked_features = features * valid.unsqueeze(-1).to(features.dtype)
        try:
            outputs = self.external_encoder(masked_features, valid)
        except TypeError as error:
            raise ValueError("external_encoder must implement forward(features, mask)") from error
        if not isinstance(outputs, torch.Tensor) or outputs.shape != (
            features.shape[0],
            features.shape[1],
            self.feature_contract.output_feature_size,
        ):
            raise ValueError(
                "external_encoder must return [batch, tokens, contract.output_feature_size]"
            )
        return NeuralEncoderOutput(
            features=outputs * valid.unsqueeze(-1).to(outputs.dtype),
            mask=valid,
            timing=timing,
            stride_samples=1,
        )
