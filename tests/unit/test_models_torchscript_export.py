import json

import torch

from openthought2text.models import (
    NeuralToTextModelConfig,
    TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE,
    build_neural_to_text_model,
    export_neural_encoder_evidence_torchscript,
    validate_neural_encoder_evidence_torchscript,
)


def _model():
    return build_neural_to_text_model(
        NeuralToTextModelConfig(
            hidden_size=8,
            temporal_kernel=5,
            stride_samples=2,
            encoder_layers=1,
            encoder_heads=2,
            encoder_dropout=0,
            vocabulary_size=11,
            decoder_layers=1,
            decoder_heads=2,
            decoder_dropout=0,
            max_sequence_length=8,
        )
    )


def _inputs():
    return (
        torch.randn(1, 2, 12),
        torch.tensor([[True] * 10 + [False] * 2]),
        torch.tensor([[True, True]]),
        torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
    )


def test_neural_encoder_torchscript_validation_traces_target_free_evidence_scope():
    model = _model()
    inputs = _inputs()
    validation = validate_neural_encoder_evidence_torchscript(model, *inputs)
    assert validation.exportable, validation.errors
    assert validation.scope == TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE
    assert validation.scripted_module is not None and validation.architecture_metadata is not None
    assert validation.input_signature == {
        "signals": {"shape": [1, 2, 12], "dtype": "torch.float32"},
        "sample_mask": {"shape": [1, 12], "dtype": "torch.bool"},
        "channel_mask": {"shape": [1, 2], "dtype": "torch.bool"},
        "coordinates": {"shape": [1, 2, 3], "dtype": "torch.float32"},
    }
    eager = model.encoder(inputs[0], sample_mask=inputs[1], channel_mask=inputs[2], coordinates=inputs[3])
    scripted = validation.scripted_module(*inputs)
    torch.testing.assert_close(scripted[0], eager.features, rtol=1e-4, atol=1e-5)
    assert torch.equal(scripted[1], eager.mask)


def test_neural_encoder_torchscript_export_writes_scoped_architecture_sidecar(tmp_path):
    path = tmp_path / "neural_encoder_evidence.pt"
    validation = export_neural_encoder_evidence_torchscript(_model(), *_inputs(), artifact_path=path)
    assert validation.exportable and path.exists()
    sidecar = json.loads((tmp_path / "neural_encoder_evidence.pt.metadata.json").read_text())
    assert sidecar["torchscript_scope"] == TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE
    assert sidecar["architecture_metadata"]["architecture_fingerprint"] == validation.architecture_metadata[
        "architecture_fingerprint"
    ]
    assert sidecar["input_signature"]["signals"]["shape"] == [1, 2, 12]
