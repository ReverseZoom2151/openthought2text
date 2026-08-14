import torch

from openthought2text.models import (
    ContinuousNeuralEncoder,
    CoordinateChannelMerger,
    NeuralTokenizerConfig,
    NeuralVectorQuantizer,
    SubjectAdapter,
)


def test_channel_merger_ignores_values_and_coordinates_of_padded_channels():
    torch.manual_seed(7)
    merger = CoordinateChannelMerger(8).eval()
    real = torch.randn(2, 3, 5, 8)
    coords = torch.randn(2, 3, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    expected = merger(real, mask, coords)
    padded = torch.cat([real, torch.randn(2, 2, 5, 8) * 1000], dim=1)
    padded_coords = torch.cat([coords, torch.randn(2, 2, 3) * 1000], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    actual = merger(padded, padded_mask, padded_coords)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_continuous_encoder_masks_padding_and_backpropagates():
    torch.manual_seed(3)
    encoder = ContinuousNeuralEncoder(hidden_size=16, num_heads=4, num_layers=1, dropout=0).eval()
    signals = torch.randn(2, 3, 31, requires_grad=True)
    sample_mask = torch.tensor([[1] * 28 + [0] * 3, [1] * 31], dtype=torch.bool)
    output = encoder(signals, sample_mask=sample_mask, sample_rate_hz=100)
    assert output.features.shape == (2, 8, 16)
    assert output.mask.shape == (2, 8)
    assert output.timing.start.shape == output.mask.shape
    assert torch.all(output.features[~output.mask] == 0)
    output.features.square().mean().backward()
    assert signals.grad is not None and signals.grad.abs().sum() > 0


def test_encoder_is_invariant_to_appended_masked_channels():
    torch.manual_seed(11)
    encoder = ContinuousNeuralEncoder(hidden_size=16, num_heads=4, num_layers=1, dropout=0).eval()
    signals = torch.randn(1, 2, 24)
    coordinates = torch.randn(1, 2, 3)
    expected = encoder(signals, coordinates=coordinates).features
    padded_signals = torch.cat([signals, torch.randn(1, 2, 24) * 100], dim=1)
    padded_coordinates = torch.cat([coordinates, torch.randn(1, 2, 3) * 100], dim=1)
    padded_mask = torch.tensor([[True, True, False, False]])
    actual = encoder(
        padded_signals, channel_mask=padded_mask, coordinates=padded_coordinates
    ).features
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_subject_adapters_have_safe_identity_initialization():
    features = torch.randn(3, 4, 6)
    ids = torch.tensor([0, 1, 2])
    for mode in ("identity", "additive", "film"):
        adapter = SubjectAdapter(6, 3, mode=mode)
        torch.testing.assert_close(adapter(features, ids), features)


def test_vector_quantizer_reports_health_masks_and_gradients():
    quantizer = NeuralVectorQuantizer(NeuralTokenizerConfig(codebook_size=4, embedding_dim=3))
    embeddings = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    output = quantizer(embeddings, mask)
    assert output.indices.shape == (2, 4)
    assert output.quantized.shape == embeddings.shape
    assert output.health.usage.shape == (4,)
    assert output.health.active_fraction + output.health.dead_fraction == 1
    assert torch.all(output.quantized[~mask] == 0)
    (output.quantized.square().mean() + output.loss).backward()
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0
