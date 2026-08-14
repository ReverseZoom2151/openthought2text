"""Run a local neural-signal-to-text smoke path without participant data.

This example is deliberately a mechanics check, not a benchmark: random
signals and random target IDs let contributors verify the protected boundary
between teacher-forced training and target-free generation.  Generated IDs
have no semantic meaning until a real, documented training run is completed.
"""

from __future__ import annotations

import torch

from openthought2text.data import NeuralTensorBatch
from openthought2text.models import (
    ContinuousNeuralEncoder,
    DecoderGenerationConfig,
    NeuralToTextModel,
    TargetFreeAutoregressiveDecoder,
)
from openthought2text.training import seed_everything, supervised_train_step


def main() -> None:
    seed_everything(7)
    batch_size, channels, samples = 2, 4, 32
    channel_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    time_mask = torch.tensor([[True] * samples, [True] * 28 + [False] * 4], dtype=torch.bool)
    signals = torch.randn(batch_size, channels, samples)
    signals *= (channel_mask.unsqueeze(-1) & time_mask.unsqueeze(1)).to(signals.dtype)
    batch = NeuralTensorBatch(
        signals=signals,
        channel_mask=channel_mask,
        time_mask=time_mask,
        sample_ids=("synthetic-000", "synthetic-001"),
        splits=("train", "train"),
    )
    encoder = ContinuousNeuralEncoder(
        hidden_size=32, temporal_kernel=5, stride_samples=4, num_layers=1, num_heads=4
    )
    decoder = TargetFreeAutoregressiveDecoder(
        vocab_size=32, hidden_size=32, num_layers=1, num_heads=4, max_sequence_length=8
    )
    model = NeuralToTextModel(encoder, decoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Text IDs are permitted only by the training API.
    target_ids = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    step = supervised_train_step(model, batch, target_ids, optimizer=optimizer)

    # No targets/labels can be supplied to target-free inference.
    model.eval()
    generated = model.generate(
        batch.signals,
        channel_mask=batch.channel_mask,
        token_mask=batch.time_mask,
        config=DecoderGenerationConfig(max_new_tokens=4),
    )
    print(
        {
            "training_loss": round(step.loss, 4),
            "generated_token_ids": generated.token_ids.tolist(),
            "neural_evidence_shape": list(generated.neural_features.shape),
            "valid_neural_tokens": generated.neural_mask.sum(dim=1).tolist(),
        }
    )


if __name__ == "__main__":
    main()
