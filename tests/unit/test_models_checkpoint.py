import pytest

from openthought2text.models import (
    NeuralToTextModelConfig,
    build_neural_to_text_model,
    checkpoint_architecture_metadata,
    validate_checkpoint_architecture,
)


def _model(hidden_size: int = 12):
    return build_neural_to_text_model(
        NeuralToTextModelConfig(
            hidden_size=hidden_size,
            temporal_kernel=5,
            stride_samples=2,
            encoder_layers=1,
            encoder_heads=3 if hidden_size == 12 else 4,
            encoder_dropout=0,
            vocabulary_size=17,
            decoder_layers=1,
            decoder_heads=3 if hidden_size == 12 else 4,
            decoder_dropout=0,
            max_sequence_length=12,
        )
    )


def test_checkpoint_metadata_matches_same_factory_architecture():
    model = _model()
    metadata = checkpoint_architecture_metadata(model)
    result = validate_checkpoint_architecture(model, metadata)
    assert result.compatible
    assert result.errors == ()
    assert result.observed_fingerprint == result.expected_fingerprint
    result.raise_if_incompatible()


def test_checkpoint_validator_reports_state_dict_incompatible_architecture():
    saved_metadata = checkpoint_architecture_metadata(_model())
    result = validate_checkpoint_architecture(_model(hidden_size=16), saved_metadata)
    assert not result.compatible
    assert "architecture fingerprint differs" in result.errors
    assert any(error.startswith("config.hidden_size differs") for error in result.errors)
    assert "state_dict schema differs" in result.errors
    with pytest.raises(ValueError, match="incompatible"):
        result.raise_if_incompatible()


def test_checkpoint_validator_rejects_malformed_or_self_inconsistent_metadata_without_loading():
    model = _model()
    malformed = validate_checkpoint_architecture(model, {"architecture_fingerprint": 3})
    assert not malformed.compatible
    assert any("fingerprint must be a string" in error for error in malformed.errors)
    assert any("description must be a mapping" in error for error in malformed.errors)
    metadata = checkpoint_architecture_metadata(model)
    metadata["architecture_fingerprint"] = "not-the-description-fingerprint"
    inconsistent = validate_checkpoint_architecture(model, metadata)
    assert not inconsistent.compatible
    assert "checkpoint fingerprint does not match its architecture_description" in inconsistent.errors
