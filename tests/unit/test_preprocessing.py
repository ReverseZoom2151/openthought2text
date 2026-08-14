from __future__ import annotations

import torch

from openthought2text.montages import validate_channel_geometry
from openthought2text.preprocessing import chunk_continuous_signal, robust_channel_scale


def test_continuous_chunking_uses_fixed_signal_windows() -> None:
    signal = torch.arange(30, dtype=torch.float32).reshape(2, 15)
    chunks = chunk_continuous_signal(signal, window_samples=6, stride_samples=4)
    assert chunks.values.shape == (3, 2, 6)
    assert chunks.start_samples.tolist() == [0, 4, 8]
    assert chunks.end_samples.tolist() == [6, 10, 14]


def test_robust_channel_scaling_and_geometry_validation() -> None:
    signal = torch.tensor([[1.0, 2.0, 3.0, 100.0], [3.0, 4.0, 5.0, 6.0]])
    scaled = robust_channel_scale(signal)
    assert torch.isfinite(scaled).all()
    validate_channel_geometry(
        ("Fz", "Cz"), torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), torch.tensor([True, True])
    )
