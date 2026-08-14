"""Strict, portable sensor geometry contracts for heterogeneous neural layouts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

import torch


class SensorType(str, Enum):
    EEG = "eeg"
    MEG = "meg"
    INTRACORTICAL = "intracortical"


_TYPE_IDS = {SensorType.EEG: 1, SensorType.MEG: 2, SensorType.INTRACORTICAL: 3}


def _vector(
    value: object, name: str, *, optional: bool = False
) -> tuple[float, float, float] | None:
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element vector")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SensorDefinition:
    name: str
    position: tuple[float, float, float]
    sensor_type: SensorType
    orientation: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("sensor name must be non-empty")
        object.__setattr__(self, "position", _vector(self.position, "position"))
        object.__setattr__(
            self, "orientation", _vector(self.orientation, "orientation", optional=True)
        )
        if not isinstance(self.sensor_type, SensorType):
            raise ValueError("sensor_type must be a SensorType")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "position": list(self.position),
            "sensor_type": self.sensor_type.value,
        }
        if self.orientation is not None:
            data["orientation"] = list(self.orientation)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SensorDefinition:
        try:
            return cls(
                name=str(data["name"]),
                position=_vector(data["position"], "position"),
                orientation=_vector(data.get("orientation"), "orientation", optional=True),
                sensor_type=SensorType(data["sensor_type"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid sensor definition") from error


@dataclass(frozen=True, slots=True)
class SensorLayout:
    """Ordered sensor geometry; order is the only tensor channel mapping."""

    layout_id: str
    sensors: tuple[SensorDefinition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id.strip():
            raise ValueError("layout_id must be non-empty")
        if not self.sensors:
            raise ValueError("sensor layout needs at least one sensor")
        names = [sensor.name for sensor in self.sensors]
        if len(names) != len(set(names)):
            raise ValueError("sensor layout names must be unique and ordered")

    def to_dict(self) -> dict[str, object]:
        return {
            "layout_id": self.layout_id,
            "sensors": [sensor.to_dict() for sensor in self.sensors],
        }

    @property
    def checksum(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SensorLayout:
        sensors = data.get("sensors")
        if not isinstance(sensors, list):
            raise ValueError("sensor layout sensors must be a list")
        return cls(
            layout_id=str(data.get("layout_id", "")),
            sensors=tuple(SensorDefinition.from_dict(item) for item in sensors),
        )


@dataclass(frozen=True, slots=True)
class SensorLayoutTensor:
    """Padding-safe tensorization with zero-valued padding in every field."""

    positions: torch.Tensor
    orientations: torch.Tensor
    orientation_mask: torch.Tensor
    sensor_type_ids: torch.Tensor
    sensor_mask: torch.Tensor
    names: tuple[str, ...]
    layout_checksum: str

    def __post_init__(self) -> None:
        size = self.positions.shape[0]
        if self.positions.shape != (size, 3) or self.orientations.shape != (size, 3):
            raise ValueError("sensor position/orientation tensors must be [sensors, 3]")
        if self.orientation_mask.shape != (size,) or self.sensor_mask.shape != (size,):
            raise ValueError("sensor masks must be [sensors]")
        if self.sensor_type_ids.shape != (size,):
            raise ValueError("sensor type IDs must be [sensors]")
        if self.orientation_mask.dtype != torch.bool or self.sensor_mask.dtype != torch.bool:
            raise ValueError("sensor masks must be boolean")
        padded = ~self.sensor_mask
        if torch.any(self.positions[padded] != 0) or torch.any(self.orientations[padded] != 0):
            raise ValueError("sensor tensor padding must be zero")
        if torch.any(self.sensor_type_ids[padded] != 0) or torch.any(self.orientation_mask[padded]):
            raise ValueError("sensor tensor padding metadata must be zero/false")


def sensor_layout_to_tensor(
    layout: SensorLayout, *, max_sensors: int | None = None
) -> SensorLayoutTensor:
    """Tensorize ordered layout geometry, leaving every padded slot inert."""
    count = len(layout.sensors)
    size = count if max_sensors is None else max_sensors
    if size < count:
        raise ValueError("max_sensors cannot be smaller than the sensor layout")
    positions = torch.zeros((size, 3), dtype=torch.float32)
    orientations = torch.zeros((size, 3), dtype=torch.float32)
    orientation_mask = torch.zeros(size, dtype=torch.bool)
    sensor_type_ids = torch.zeros(size, dtype=torch.long)
    sensor_mask = torch.zeros(size, dtype=torch.bool)
    for index, sensor in enumerate(layout.sensors):
        positions[index] = torch.tensor(sensor.position, dtype=torch.float32)
        sensor_type_ids[index] = _TYPE_IDS[sensor.sensor_type]
        sensor_mask[index] = True
        if sensor.orientation is not None:
            orientations[index] = torch.tensor(sensor.orientation, dtype=torch.float32)
            orientation_mask[index] = True
    return SensorLayoutTensor(
        positions=positions,
        orientations=orientations,
        orientation_mask=orientation_mask,
        sensor_type_ids=sensor_type_ids,
        sensor_mask=sensor_mask,
        names=tuple(sensor.name for sensor in layout.sensors),
        layout_checksum=layout.checksum,
    )
