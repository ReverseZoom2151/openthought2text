from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from openthought2text.data import NamedMontage, SignalReference, TensorBackedSample, select_named_montage

from .test_data_schema import sample


def row():
    base = sample()
    base = replace(base, signal=SignalReference(
        uri=base.signal.uri, recording_id=base.signal.recording_id,
        sampling_rate_hz=base.signal.sampling_rate_hz, channel_count=3,
        channel_names=("Fz", "Cz", "Pz"),
    ))
    return TensorBackedSample(
        sample=base,
        values=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        channel_mask=torch.tensor([True, False, True]),
        time_mask=torch.tensor([True, True]),
    )


def montage():
    return NamedMontage(
        name="central_then_frontal",
        channel_names=("Cz", "Fz"),
        coordinates={"Fz": (0.0, 1.0, 0.0), "Cz": (0.0, 0.0, 0.0)},
    )


def test_named_montage_selects_declared_order_preserves_masks_and_records_provenance() -> None:
    rows, provenance = select_named_montage((row(),), montage())

    assert rows[0].sample.signal.channel_names == ("Cz", "Fz")
    assert rows[0].values.tolist() == [[3.0, 4.0], [1.0, 2.0]]
    assert rows[0].channel_mask.tolist() == [False, True]
    assert provenance.source_channel_counts == (3,)
    assert provenance.montage.to_dict()["channel_names"] == ["Cz", "Fz"]


def test_named_montage_rejects_missing_names_and_incomplete_coordinates() -> None:
    with pytest.raises(ValueError, match="map exactly"):
        NamedMontage(name="bad", channel_names=("Cz",), coordinates={})
    missing = NamedMontage(name="missing", channel_names=("Oz",), coordinates={"Oz": (0, 0, 0)})
    with pytest.raises(ValueError, match="lacks montage channels"):
        select_named_montage((row(),), missing)
