import inspect

import torch

from openthought2text.models import (
    ContinuousNeuralEncoder,
    DecoderGenerationConfig,
    NeuralToTextModel,
    SemanticAnchorHead,
    TargetFreeAutoregressiveDecoder,
)


def _model() -> NeuralToTextModel:
    return NeuralToTextModel(
        encoder=ContinuousNeuralEncoder(hidden_size=12, num_heads=3, num_layers=1, stride_samples=4, dropout=0),
        decoder=TargetFreeAutoregressiveDecoder(
            vocab_size=11, hidden_size=12, num_heads=3, num_layers=1, max_sequence_length=10, dropout=0
        ),
        semantic_anchor_head=SemanticAnchorHead(hidden_size=12, num_anchors=4),
    ).eval()


def test_composed_training_forward_connects_encoder_decoder_and_anchors():
    torch.manual_seed(2)
    model = _model()
    signals = torch.randn(2, 3, 21, requires_grad=True)
    sample_mask = torch.tensor([[True] * 19 + [False] * 2, [True] * 21])
    target_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    # 21 samples at stride four produces six neural tokens.
    anchor_targets = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]])
    output = model(signals, target_ids, token_mask=sample_mask, anchor_targets=anchor_targets)
    assert output.encoder.features.shape == (2, 6, 12)
    assert output.decoder.logits.shape == (2, 3, 11)
    assert output.anchors is not None and output.anchors.logits.shape == (2, 6, 4)
    output.loss.backward()
    assert signals.grad is not None and signals.grad.abs().sum() > 0


def test_composed_generate_is_target_free_and_returns_evidence():
    torch.manual_seed(5)
    model = _model()
    signals = torch.randn(1, 2, 20)
    signature = inspect.signature(model.generate)
    assert set(signature.parameters) == {
        "signals", "channel_mask", "coordinates", "token_mask", "config", "sample_rate_hz"
    }
    result = model.generate(
        signals,
        channel_mask=torch.tensor([[True, True]]),
        token_mask=torch.tensor([[True, True, True, False, False]]),
        config=DecoderGenerationConfig(max_new_tokens=3),
        sample_rate_hz=100,
    )
    assert result.token_ids.shape == (1, 3)
    assert result.neural_features.shape == (1, 5, 12)
    assert result.neural_mask.tolist() == [[True, True, True, False, False]]
    assert torch.all(result.neural_features[:, 3:] == 0)
    assert result.timing.sample_rate_hz == 100
    assert result.anchor_logits is not None and result.anchor_logits.shape == (1, 5, 4)


def test_composed_generate_uses_sample_mask_when_mask_matches_signal_time():
    model = _model()
    signals = torch.randn(1, 2, 20)
    result = model.generate(
        signals,
        token_mask=torch.tensor([[True] * 16 + [False] * 4]),
        config=DecoderGenerationConfig(max_new_tokens=2),
    )
    assert result.neural_mask.tolist() == [[True, True, True, True, False]]
