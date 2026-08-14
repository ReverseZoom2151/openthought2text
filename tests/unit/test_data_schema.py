from openthought2text.data import (
    InformationAccess,
    Modality,
    NeuralTextSample,
    SignalReference,
    TextTarget,
    TimeInterval,
)
from openthought2text.data.schema import SchemaError
import pytest


def sample(**changes):
    values = dict(
        sample_id="s-1", dataset_id="zuco", subject_id="sub-1", modality=Modality.EEG,
        signal=SignalReference("signals/sub-1.npz", "recording-1", 250, 2, channel_names=("Fz", "Cz")),
        interval=TimeInterval(0.0, 1.0), target=TextTarget("A careful sentence."),
        split="train", group_ids=("subject:sub-1",), task="reading",
    )
    values.update(changes)
    return NeuralTextSample(**values)


def test_sample_round_trip_preserves_contract():
    restored = NeuralTextSample.from_dict(sample().to_dict())
    assert restored == sample()
    assert restored.target and restored.target.fingerprint


def test_invalid_channel_metadata_is_rejected():
    with pytest.raises(SchemaError, match="channel_names"):
        SignalReference("x", "r", 100, 2, channel_names=("Fz",))


def test_information_access_explicitly_flags_inference_text():
    access = InformationAccess(inference_text_context=True)
    assert access.inference_label_leakage
