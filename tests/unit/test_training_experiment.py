from __future__ import annotations

import torch

from openthought2text.data import TensorBackedSample, collate_tensor_backed_samples
from openthought2text.models import NeuralToTextModelConfig, build_neural_to_text_model
from openthought2text.training import build_training_inputs, target_ids_for_batch, train_one_epoch

from .test_data_schema import sample


def _row(sample_id: str, text: str, split: str = "train") -> TensorBackedSample:
    return TensorBackedSample(
        sample(sample_id=sample_id, split=split, target=sample().target.__class__(text)),
        torch.arange(16, dtype=torch.float32).reshape(2, 8),
    )


def _model(vocabulary_size: int):
    return build_neural_to_text_model(
        NeuralToTextModelConfig(
            hidden_size=16,
            temporal_kernel=3,
            stride_samples=2,
            encoder_layers=1,
            decoder_layers=1,
            encoder_heads=4,
            decoder_heads=4,
            vocabulary_size=vocabulary_size,
            max_sequence_length=16,
            encoder_dropout=0.0,
            decoder_dropout=0.0,
        )
    )


def test_training_inputs_and_one_epoch_are_train_only() -> None:
    inputs = build_training_inputs((_row("a", "alpha beta"), _row("b", "beta gamma")))
    model = _model(len(inputs.tokenizer.vocabulary))
    results = train_one_epoch(
        model,
        inputs.rows,
        inputs.tokenizer,
        optimizer=torch.optim.AdamW(model.parameters()),
        batch_size=1,
    )
    assert len(results) == 2
    assert all(result.loss > 0 for result in results)


def test_target_ids_reject_nontrain_source() -> None:
    train = _row("a", "alpha")
    inputs = build_training_inputs((train,))
    batch = collate_tensor_backed_samples((train,))
    bad = _row("a", "alpha", split="test")
    try:
        target_ids_for_batch(batch, {"a": bad}, inputs.tokenizer)
    except ValueError as error:
        assert "not authorized" in str(error)
    else:
        raise AssertionError("non-train target unexpectedly accepted")
