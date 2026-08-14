from __future__ import annotations

import pytest
import torch

from openthought2text.data import (
    SensorDefinition,
    SensorLayout,
    SensorType,
    sensor_layout_to_tensor,
)


def layout() -> SensorLayout:
    return SensorLayout(
        layout_id="mixed-fixture",
        sensors=(
            SensorDefinition("Fz", (0, 1, 0), SensorType.EEG),
            SensorDefinition("MEG011", (1, 0, 0), SensorType.MEG, (0, 0, 1)),
            SensorDefinition("array-1", (0, 0, 1), SensorType.INTRACORTICAL),
        ),
    )


def test_sensor_layout_is_ordered_checksummed_and_optional_orientation_is_explicit() -> None:
    value = layout()
    restored = SensorLayout.from_dict(value.to_dict())
    assert restored == value
    assert restored.checksum == value.checksum
    assert sensor_layout_to_tensor(value).orientation_mask.tolist() == [False, True, False]


def test_sensor_layout_rejects_missing_type_and_invalid_position() -> None:
    with pytest.raises(ValueError, match="invalid sensor definition"):
        SensorDefinition.from_dict({"name": "Fz", "position": [0, 0, 0]})
    with pytest.raises(ValueError, match="finite"):
        SensorDefinition("Fz", (0, float("nan"), 0), SensorType.EEG)


def test_sensor_layout_tensor_padding_is_zero_and_invariant_to_padded_values() -> None:
    tensor = sensor_layout_to_tensor(layout(), max_sensors=5)
    assert tensor.sensor_mask.tolist() == [True, True, True, False, False]
    assert torch.equal(tensor.positions[3:], torch.zeros((2, 3)))
    assert tensor.sensor_type_ids[3:].tolist() == [0, 0]
    # The contract exposes no values in padded slots, so tensorized geometry is
    # invariant to any hypothetical padded-channel payload values.
    assert torch.equal(tensor.orientations[3:], torch.zeros((2, 3)))
