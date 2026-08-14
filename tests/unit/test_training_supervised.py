from __future__ import annotations

import torch

from openthought2text.data import NeuralTensorBatch
from openthought2text.models import (
    ContinuousNeuralEncoder,
    NeuralToTextModel,
    TargetFreeAutoregressiveDecoder,
)
from openthought2text.training import supervised_train_step


def test_supervised_step_updates_composed_model() -> None:
    encoder = ContinuousNeuralEncoder(hidden_size=16, num_layers=1, num_heads=4, stride_samples=2)
    decoder = TargetFreeAutoregressiveDecoder(
        vocab_size=12, hidden_size=16, num_layers=1, num_heads=4
    )
    model = NeuralToTextModel(encoder, decoder)
    batch = NeuralTensorBatch(
        signals=torch.randn(2, 2, 8),
        channel_mask=torch.ones(2, 2, dtype=torch.bool),
        time_mask=torch.ones(2, 8, dtype=torch.bool),
        sample_ids=("a", "b"),
        splits=("train", "train"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = supervised_train_step(
        model, batch, torch.tensor([[1, 2, 3], [2, 3, 4]]), optimizer=optimizer
    )
    assert result.loss > 0
    assert result.batch_size == 2
    assert result.gradient_norm is not None
