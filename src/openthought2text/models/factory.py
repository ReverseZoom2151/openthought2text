"""Strict construction and checkpoint-compatible description for tiny models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import torch
from torch.nn.parameter import UninitializedParameter

from .decoder import TargetFreeAutoregressiveDecoder
from .encoder import ContinuousNeuralEncoder
from .heads import SemanticAnchorHead
from .model import NeuralToTextModel


@dataclass(frozen=True)
class NeuralToTextModelConfig:
    """Validated architecture-only configuration for the initial tiny model."""

    hidden_size: int = 128
    temporal_kernel: int = 9
    stride_samples: int = 4
    encoder_layers: int = 2
    encoder_heads: int = 4
    encoder_dropout: float = 0.1
    coordinate_size: int = 3
    vocabulary_size: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_dropout: float = 0.1
    max_sequence_length: int = 128
    bos_token_id: int = 0
    pad_token_id: int = 0
    semantic_anchor_classes: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "temporal_kernel": self.temporal_kernel,
            "stride_samples": self.stride_samples,
            "encoder_layers": self.encoder_layers,
            "encoder_heads": self.encoder_heads,
            "coordinate_size": self.coordinate_size,
            "vocabulary_size": self.vocabulary_size,
            "decoder_layers": self.decoder_layers,
            "decoder_heads": self.decoder_heads,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        if self.hidden_size % self.encoder_heads or self.hidden_size % self.decoder_heads:
            raise ValueError("hidden_size must divide evenly by encoder_heads and decoder_heads")
        for name, value in {"encoder_dropout": self.encoder_dropout, "decoder_dropout": self.decoder_dropout}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be a number in [0, 1)")
        for name, token_id in {"bos_token_id": self.bos_token_id, "pad_token_id": self.pad_token_id}.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < self.vocabulary_size:
                raise ValueError(f"{name} must be a vocabulary ID")
        if self.semantic_anchor_classes is not None and (
            isinstance(self.semantic_anchor_classes, bool)
            or not isinstance(self.semantic_anchor_classes, int)
            or self.semantic_anchor_classes < 2
        ):
            raise ValueError("semantic_anchor_classes must be None or an integer of at least two")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NeuralToTextModelConfig":
        """Accept only declared configuration fields; reject silent misspellings."""
        if not isinstance(values, Mapping):
            raise ValueError("model configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown model configuration fields: {sorted(unknown)}")
        return cls(**dict(values))


def _schema_for(model: NeuralToTextModel, config: NeuralToTextModelConfig) -> list[dict[str, Any]]:
    """State-dict key/shape/schema description, including the lazy temporal conv."""
    schema: list[dict[str, Any]] = []
    for name, value in sorted(model.state_dict().items()):
        if isinstance(value, UninitializedParameter):
            if name.endswith("encoder.temporal.0.weight"):
                shape = [config.hidden_size, 1, config.temporal_kernel]
            elif name.endswith("encoder.temporal.0.bias"):
                shape = [config.hidden_size]
            else:  # defensive: this factory has no other lazy parameters.
                raise RuntimeError(f"unrecognized uninitialized state-dict parameter: {name}")
            dtype = "torch.float32"
        else:
            shape = list(value.shape)
            dtype = str(value.dtype)
        schema.append({"name": name, "shape": shape, "dtype": dtype})
    return schema


def describe_model_architecture(model: NeuralToTextModel) -> dict[str, Any]:
    """Return a JSON-serializable state-dict-compatible architecture descriptor."""
    config = getattr(model, "_architecture_config", None)
    if not isinstance(config, NeuralToTextModelConfig):
        raise ValueError("model was not built by build_neural_to_text_model")
    return {
        "format_version": 1,
        "model_type": "NeuralToTextModel",
        "config": asdict(config),
        "state_dict_schema": _schema_for(model, config),
    }


def architecture_fingerprint(model: NeuralToTextModel) -> str:
    """Stable SHA-256 fingerprint for architecture and state-dict compatibility."""
    canonical = json.dumps(describe_model_architecture(model), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def build_neural_to_text_model(
    config: NeuralToTextModelConfig | Mapping[str, Any],
) -> NeuralToTextModel:
    """Build the current continuous encoder + target-free decoder architecture."""
    if isinstance(config, Mapping):
        config = NeuralToTextModelConfig.from_mapping(config)
    if not isinstance(config, NeuralToTextModelConfig):
        raise ValueError("config must be NeuralToTextModelConfig or a mapping")
    encoder = ContinuousNeuralEncoder(
        hidden_size=config.hidden_size,
        temporal_kernel=config.temporal_kernel,
        stride_samples=config.stride_samples,
        num_layers=config.encoder_layers,
        num_heads=config.encoder_heads,
        dropout=float(config.encoder_dropout),
        coordinate_size=config.coordinate_size,
    )
    decoder = TargetFreeAutoregressiveDecoder(
        vocab_size=config.vocabulary_size,
        hidden_size=config.hidden_size,
        num_layers=config.decoder_layers,
        num_heads=config.decoder_heads,
        max_sequence_length=config.max_sequence_length,
        bos_token_id=config.bos_token_id,
        pad_token_id=config.pad_token_id,
        dropout=float(config.decoder_dropout),
    )
    anchors = (
        SemanticAnchorHead(config.hidden_size, config.semantic_anchor_classes)
        if config.semantic_anchor_classes is not None
        else None
    )
    model = NeuralToTextModel(encoder=encoder, decoder=decoder, semantic_anchor_head=anchors)
    # Store immutable config only; description/fingerprint are recomputed so
    # callers see the same result before and after lazy-module materialization.
    model._architecture_config = config  # type: ignore[attr-defined]
    return model
