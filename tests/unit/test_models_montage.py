import torch

from openthought2text.models import GraphMontageAdapter


def test_graph_montage_shapes_gradients_and_missing_channel_zeroing():
    torch.manual_seed(4)
    adapter = GraphMontageAdapter(hidden_size=6)
    features = torch.randn(2, 3, 4, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]).expand(2, -1, -1)
    output = adapter(features, mask, coordinates)
    assert output.shape == features.shape
    assert torch.all(output[0, 2] == 0)
    assert torch.all(output[1, 1] == 0)
    output.square().mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_graph_montage_valid_outputs_ignore_appended_padded_values_and_coordinates():
    torch.manual_seed(12)
    adapter = GraphMontageAdapter(hidden_size=4).eval()
    real = torch.randn(1, 2, 3, 4)
    real_coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    expected = adapter(real, torch.tensor([[True, True]]), real_coords)
    padded = torch.cat([real, torch.randn(1, 2, 3, 4) * 10_000], dim=1)
    padded_coords = torch.cat([real_coords, torch.randn(1, 2, 3) * 10_000], dim=1)
    actual = adapter(padded, torch.tensor([[True, True, False, False]]), padded_coords)
    torch.testing.assert_close(actual[:, :2], expected, rtol=1e-6, atol=1e-6)
    assert torch.all(actual[:, 2:] == 0)


def test_graph_weights_support_one_available_channel_and_normalize_valid_rows():
    adapter = GraphMontageAdapter(hidden_size=3)
    mask = torch.tensor([[False, True, False]])
    coords = torch.tensor([[[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    weights = adapter.graph_weights(mask, coords)
    assert weights.shape == (1, 3, 3)
    assert weights[0, 1].tolist() == [0.0, 1.0, 0.0]
    assert torch.isfinite(weights).all()
