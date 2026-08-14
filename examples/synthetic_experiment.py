"""A complete local experiment trace over the non-participant synthetic fixture.

The output is intentionally not a benchmark.  It demonstrates the artifact
boundaries a real run must respect: train-only normalization and vocabulary,
teacher-forced training, target-free evaluation generation, and checkpoint
metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from openthought2text.config.run import RunManifest
from openthought2text.data import (
    SyntheticNeuralTextAdapter,
    TensorBackedSample,
    collate_tensor_backed_samples,
    fit_train_channel_normalizer,
    fit_train_text_tokenizer,
)
from openthought2text.models import NeuralToTextModelConfig, build_neural_to_text_model
from openthought2text.training import CheckpointMetadata, save_checkpoint, seed_everything, supervised_train_step


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _targets(rows: tuple[TensorBackedSample, ...], tokenizer) -> torch.Tensor:
    encoded = [tokenizer.encode(row.sample.target.text) for row in rows]
    result = torch.full((len(encoded), max(map(len, encoded))), -100, dtype=torch.long)
    for index, sequence in enumerate(encoded):
        result[index, : len(sequence)] = torch.tensor(sequence)
    return result


def main() -> None:
    seed_everything(7)
    with TemporaryDirectory(prefix="openthought2text-example-") as directory:
        root = Path(directory)
        manifest = SyntheticNeuralTextAdapter().generate(str(root))
        prepared = []
        for sample in manifest.samples:
            values = torch.tensor(json.loads(Path(sample.signal.uri).read_text()), dtype=torch.float32)
            prepared.append(TensorBackedSample(sample, values))
        train = tuple(row for row in prepared if row.sample.split == "train")
        held_out = tuple(row for row in prepared if row.sample.split != "train")
        normalizer = fit_train_channel_normalizer(train)
        tokenizer = fit_train_text_tokenizer(
            (row.sample for row in train), unknown_policy="unk"
        )
        normalized_train = tuple(TensorBackedSample(row.sample, normalizer.apply(row.values)) for row in train)
        train_batch = collate_tensor_backed_samples(normalized_train)
        config = NeuralToTextModelConfig(
            vocabulary_size=len(tokenizer.vocabulary), hidden_size=32, temporal_kernel=5,
            stride_samples=4, encoder_layers=1, decoder_layers=1, encoder_heads=4,
            decoder_heads=4, max_sequence_length=32, encoder_dropout=0.0, decoder_dropout=0.0,
        )
        model = build_neural_to_text_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        step = supervised_train_step(model, train_batch, _targets(normalized_train, tokenizer), optimizer=optimizer)

        evaluation_batch = collate_tensor_backed_samples(held_out)
        model.eval()
        generated = model.generate(
            evaluation_batch.signals, channel_mask=evaluation_batch.channel_mask,
            token_mask=evaluation_batch.time_mask,
        )
        checkpoint = root / "checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model=model,
            metadata=CheckpointMetadata(
                epoch=0, step=1, selection_metric="synthetic_train_loss", selection_value=step.loss,
                run_manifest=RunManifest(
                    experiment_name="synthetic_trace", dataset_artifact_checksum=_digest(root / "synthetic_manifest.jsonl"),
                    split_manifest_checksum=_digest(root / "synthetic_manifest.jsonl"), seed=7,
                    resolved_config={"model": asdict(config), "tokenizer_checksum": tokenizer.checksum,
                                     "normalizer_checksum": normalizer.checksum},
                ),
            ), optimizer=optimizer,
        )
        print({"train_loss": round(step.loss, 4), "held_out_ids": evaluation_batch.sample_ids,
               "generated_token_ids": generated.token_ids.tolist(), "checkpoint_sha256": _digest(checkpoint)})


if __name__ == "__main__":
    main()
