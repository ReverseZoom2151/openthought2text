import pytest
import torch

from openthought2text.models import (
    ChannelNetNeuralEncoder,
    CompactConformerNeuralEncoder,
    GRUNeuralEncoder,
)


def _encoders():
    return [
        ChannelNetNeuralEncoder(hidden_size=8, temporal_kernel=5, stride_samples=4),
        GRUNeuralEncoder(hidden_size=8, temporal_kernel=5, stride_samples=4),
        CompactConformerNeuralEncoder(
            hidden_size=8, temporal_kernel=5, stride_samples=4, num_layers=1, num_heads=2, dropout=0
        ),
    ]


@pytest.mark.parametrize("encoder", _encoders())
def test_encoder_baselines_shapes_masks_timing_and_gradients(encoder):
    torch.manual_seed(27)
    encoder.eval()
    signals = torch.randn(2, 2, 17, requires_grad=True)
    sample_mask = torch.tensor([[True] * 13 + [False] * 4, [True] * 17])
    output = encoder(signals, sample_mask=sample_mask, sample_rate_hz=100)
    assert output.features.shape == (2, 5, 8)
    assert output.mask.tolist() == [[True, True, True, True, False], [True, True, True, True, True]]
    assert output.timing.start[0].tolist() == [0, 4, 8, 12, 16]
    assert output.timing.end[0].tolist() == [4, 8, 12, 16, 17]
    assert output.timing.sample_rate_hz == 100
    assert torch.all(output.features[~output.mask] == 0)
    output.features.square().mean().backward()
    assert signals.grad is not None and signals.grad.abs().sum() > 0


@pytest.mark.parametrize("encoder", _encoders())
def test_encoder_baselines_ignore_appended_masked_channels_and_coordinates(encoder):
    torch.manual_seed(28)
    encoder.eval()
    signals = torch.randn(1, 2, 16)
    coordinates = torch.randn(1, 2, 3)
    expected = encoder(signals, coordinates=coordinates).features
    padded_signals = torch.cat([signals, torch.randn(1, 2, 16) * 1_000], dim=1)
    padded_coordinates = torch.cat([coordinates, torch.randn(1, 2, 3) * 1_000], dim=1)
    actual = encoder(
        padded_signals,
        channel_mask=torch.tensor([[True, True, False, False]]),
        coordinates=padded_coordinates,
    ).features
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_gru_baseline_rejects_holey_masks_that_cannot_be_packed_safely():
    encoder = GRUNeuralEncoder(hidden_size=4, temporal_kernel=3, stride_samples=2)
    with pytest.raises(ValueError, match="prefix-valid"):
        encoder(
            torch.randn(1, 2, 8),
            sample_mask=torch.tensor([[True, True, False, False, True, True, True, True]]),
        )
