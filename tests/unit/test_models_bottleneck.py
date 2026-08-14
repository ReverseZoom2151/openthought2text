import torch

from openthought2text.models import (
    NeuralRepresentationBottleneck,
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)


def test_bottleneck_without_quantizer_is_masked_identity_with_safe_loss():
    features = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    output = NeuralRepresentationBottleneck(4)(features, mask)
    assert not output.is_quantized
    assert output.indices is None and output.per_level_health == ()
    torch.testing.assert_close(output.continuous_features, output.quantized_features)
    assert torch.all(output.features[~mask] == 0)
    output.loss.backward()
    assert features.grad is not None


def test_bottleneck_quantization_exposes_continuous_discrete_health_and_gradients():
    torch.manual_seed(17)
    quantizer = ResidualVectorQuantizer(
        ResidualVectorQuantizerConfig(num_codebooks=2, codebook_size=5, embedding_dim=4)
    )
    bottleneck = NeuralRepresentationBottleneck(4, quantizer)
    features = torch.randn(2, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    output = bottleneck(features, mask)
    assert output.is_quantized
    assert output.indices is not None and output.indices.shape == (2, 4, 2)
    assert len(output.per_level_health) == 2
    assert torch.all(output.continuous_features[~mask] == 0)
    assert torch.all(output.quantized_features[~mask] == 0)
    (output.features.square().mean() + output.loss).backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert all(book.weight.grad is not None for book in quantizer.codebooks)


def test_bottleneck_ignores_arbitrary_padded_values_in_continuous_and_vq_paths():
    torch.manual_seed(18)
    bottleneck = NeuralRepresentationBottleneck(
        3,
        ResidualVectorQuantizer(
            ResidualVectorQuantizerConfig(num_codebooks=2, codebook_size=4, embedding_dim=3)
        ),
    ).eval()
    mask = torch.tensor([[True, True, False, False]])
    original = torch.randn(1, 4, 3)
    changed = original.clone()
    changed[:, 2:] = torch.randn(1, 2, 3) * 1_000
    first = bottleneck(original, mask)
    second = bottleneck(changed, mask)
    torch.testing.assert_close(
        first.continuous_features, second.continuous_features, rtol=0, atol=0
    )
    torch.testing.assert_close(first.quantized_features, second.quantized_features, rtol=0, atol=0)
    torch.testing.assert_close(first.indices, second.indices)
    for first_health, second_health in zip(first.per_level_health, second.per_level_health):
        torch.testing.assert_close(first_health.usage, second_health.usage)
