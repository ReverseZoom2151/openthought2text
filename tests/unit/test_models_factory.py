import pytest
import torch

from openthought2text.models import (
    NeuralToTextModelConfig,
    architecture_fingerprint,
    build_neural_to_text_model,
    describe_model_architecture,
)


def _config() -> NeuralToTextModelConfig:
    return NeuralToTextModelConfig(
        hidden_size=12,
        temporal_kernel=5,
        stride_samples=2,
        encoder_layers=1,
        encoder_heads=3,
        encoder_dropout=0,
        vocabulary_size=17,
        decoder_layers=1,
        decoder_heads=3,
        decoder_dropout=0,
        max_sequence_length=12,
        semantic_anchor_classes=4,
    )


def test_factory_builds_current_architecture_and_stable_state_dict_fingerprint():
    first = build_neural_to_text_model(_config())
    second = build_neural_to_text_model(_config())
    first_description = describe_model_architecture(first)
    assert first.semantic_anchor_head is not None
    assert first_description["config"]["hidden_size"] == 12
    assert any(item["name"] == "encoder.temporal.0.weight" for item in first_description["state_dict_schema"])
    assert architecture_fingerprint(first) == architecture_fingerprint(second)
    # Materializing the lazy channel-independent conv does not alter the
    # architecture descriptor or checkpoint compatibility.
    before = architecture_fingerprint(first)
    first.encoder(torch.randn(1, 2, 9))
    second.encoder(torch.randn(1, 3, 9))
    assert architecture_fingerprint(first) == before
    second.load_state_dict(first.state_dict())


def test_factory_fingerprint_changes_for_state_dict_incompatible_architecture():
    base = build_neural_to_text_model(_config())
    changed = build_neural_to_text_model(
        NeuralToTextModelConfig(**{**_config().__dict__, "semantic_anchor_classes": None})
    )
    assert architecture_fingerprint(base) != architecture_fingerprint(changed)
    assert set(base.state_dict()) != set(changed.state_dict())


def test_factory_strict_config_validation_rejects_unknown_bad_and_incompatible_values():
    with pytest.raises(ValueError, match="unknown"):
        NeuralToTextModelConfig.from_mapping({"hidden_szie": 12})
    with pytest.raises(ValueError, match="positive integer"):
        NeuralToTextModelConfig(hidden_size=True)
    with pytest.raises(ValueError, match="divide"):
        NeuralToTextModelConfig(hidden_size=10, encoder_heads=3)
    with pytest.raises(ValueError, match="vocabulary ID"):
        NeuralToTextModelConfig(vocabulary_size=4, bos_token_id=4)
    with pytest.raises(ValueError, match="config must"):
        build_neural_to_text_model(object())
