import inspect

import torch

from openthought2text.models import DecoderGenerationConfig, TargetFreeAutoregressiveDecoder


def _decoder() -> TargetFreeAutoregressiveDecoder:
    return TargetFreeAutoregressiveDecoder(
        vocab_size=17, hidden_size=12, num_heads=3, num_layers=1, max_sequence_length=12, dropout=0
    ).eval()


def test_teacher_forced_forward_has_expected_shape_and_gradients():
    decoder = _decoder()
    neural = torch.randn(2, 5, 12, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    targets = torch.tensor([[2, 3, 4], [4, 3, 2]])
    output = decoder(neural, mask, targets)
    assert output.logits.shape == (2, 3, 17)
    assert output.loss is not None
    output.loss.backward()
    assert neural.grad is not None and neural.grad.abs().sum() > 0


def test_generate_has_no_target_or_label_argument_and_is_label_invariant():
    torch.manual_seed(9)
    decoder = _decoder()
    neural = torch.randn(2, 4, 12)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    signature = inspect.signature(decoder.generate)
    assert set(signature.parameters) == {"neural_features", "neural_mask", "config"}
    first = decoder.generate(neural, mask, DecoderGenerationConfig(max_new_tokens=4))
    # Changing labels used in a separate training forward cannot enter generate.
    decoder(neural, mask, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    second = decoder.generate(neural, mask, DecoderGenerationConfig(max_new_tokens=4))
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 4)


def test_generation_respects_eos_and_neural_mask_shape_validation():
    decoder = _decoder()
    with torch.no_grad():
        decoder.output_projection.weight.zero_()
    neural = torch.randn(1, 3, 12)
    mask = torch.ones(1, 3, dtype=torch.bool)
    emitted = decoder.generate(neural, mask, DecoderGenerationConfig(max_new_tokens=6, eos_token_id=0))
    assert emitted.shape == (1, 1)
    assert emitted.item() == 0
