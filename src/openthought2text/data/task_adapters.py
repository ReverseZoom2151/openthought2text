"""Typed discovery-only contracts for task-specific research datasets.

These adapters inspect an explicit JSON descriptor and expected local paths.
They deliberately never deserialize participant recordings or labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .manifest import DatasetManifest
from .schema import InformationAccess, Modality, NeuralTextSample


DESCRIPTOR_NAME = "task_adapter.json"
DESCRIPTOR_KIND = "openthought2text.task_dataset_descriptor"


class LabelAccess(str, Enum):
    TRAIN_ONLY = "train_only"


@dataclass(frozen=True, slots=True)
class TaskAdapterRequirements:
    name: str
    task: str
    allowed_modalities: tuple[Modality, ...]
    required_paths: tuple[tuple[str, str], ...]
    label_access: LabelAccess = LabelAccess.TRAIN_ONLY


@dataclass(frozen=True, slots=True)
class TaskDiscoveryIssue:
    code: str
    message: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class TaskDiscoveryReport:
    root: Path
    requirements: TaskAdapterRequirements
    descriptor: Mapping[str, Any] | None = None
    issues: tuple[TaskDiscoveryIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return self.descriptor is not None and not self.issues

    @property
    def errors(self) -> tuple[TaskDiscoveryIssue, ...]:
        return self.issues

    def require_valid(self) -> Mapping[str, Any]:
        if not self.passed:
            codes = ", ".join(issue.code for issue in self.issues) or "missing descriptor"
            raise ValueError(f"task dataset discovery failed: {codes}")
        assert self.descriptor is not None
        return self.descriptor


def _local_relative_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "://" in value or value.startswith("file:"):
        raise ValueError(f"{field} must be a local relative path")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the dataset root") from error
    return resolved


class TypedTaskDiscoveryAdapter:
    """Base contract for typed task discovery with no participant-data parsing."""

    requirements: TaskAdapterRequirements
    dataset_id = "task_inventory"

    @property
    def name(self) -> str:
        return self.requirements.name

    def discover(self, source: str) -> TaskDiscoveryReport:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            return TaskDiscoveryReport(
                root,
                self.requirements,
                issues=(TaskDiscoveryIssue("MISSING_DATASET_DIRECTORY", "dataset root is not a directory", root),),
            )
        descriptor_path = root / DESCRIPTOR_NAME
        if not descriptor_path.is_file():
            return TaskDiscoveryReport(
                root,
                self.requirements,
                issues=(TaskDiscoveryIssue("MISSING_TASK_DESCRIPTOR", "task_adapter.json is required", descriptor_path),),
            )
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return TaskDiscoveryReport(
                root,
                self.requirements,
                issues=(TaskDiscoveryIssue("INVALID_TASK_DESCRIPTOR_JSON", str(error), descriptor_path),),
            )
        if not isinstance(descriptor, dict):
            return TaskDiscoveryReport(
                root,
                self.requirements,
                issues=(TaskDiscoveryIssue("INVALID_TASK_DESCRIPTOR", "descriptor must be a JSON object", descriptor_path),),
            )
        issues = self._validate_descriptor(root, descriptor)
        return TaskDiscoveryReport(root, self.requirements, descriptor, tuple(issues))

    def _validate_descriptor(self, root: Path, descriptor: Mapping[str, Any]) -> list[TaskDiscoveryIssue]:
        issues: list[TaskDiscoveryIssue] = []
        if descriptor.get("kind") != DESCRIPTOR_KIND:
            issues.append(TaskDiscoveryIssue("INVALID_DESCRIPTOR_KIND", "descriptor kind is not recognized"))
        if not isinstance(descriptor.get("dataset_id"), str) or not descriptor["dataset_id"].strip():
            issues.append(TaskDiscoveryIssue("MISSING_DATASET_ID", "descriptor dataset_id is required"))
        if descriptor.get("task") != self.requirements.task:
            issues.append(TaskDiscoveryIssue("TASK_MISMATCH", f"descriptor task must be {self.requirements.task!r}"))
        try:
            modality = Modality(descriptor.get("modality"))
        except ValueError:
            issues.append(TaskDiscoveryIssue("INVALID_MODALITY", "descriptor modality is invalid"))
        else:
            if modality not in self.requirements.allowed_modalities:
                allowed = ", ".join(item.value for item in self.requirements.allowed_modalities)
                issues.append(TaskDiscoveryIssue("MODALITY_MISMATCH", f"allowed modalities: {allowed}"))
        self._validate_label_access(descriptor.get("label_access"), issues)
        self._validate_splits(descriptor.get("splits"), issues)
        self._validate_alignment(descriptor.get("alignment_source"), issues)
        for relative_path, expected_kind in self.requirements.required_paths:
            path = root / relative_path
            exists = path.is_dir() if expected_kind == "directory" else path.is_file()
            if not exists:
                issues.append(TaskDiscoveryIssue("MISSING_REQUIRED_PATH", f"expected {expected_kind}: {relative_path}", path))
        for key, expected_kind in (("recordings_dir", "directory"), ("split_manifest", "file")):
            try:
                path = _local_relative_path(root, descriptor.get(key), key)
            except ValueError as error:
                issues.append(TaskDiscoveryIssue("INVALID_LOCAL_REFERENCE", str(error)))
                continue
            exists = path.is_dir() if expected_kind == "directory" else path.is_file()
            if not exists:
                issues.append(TaskDiscoveryIssue("MISSING_DECLARED_PATH", f"descriptor {key} is not a {expected_kind}", path))
        return issues

    @staticmethod
    def _validate_label_access(value: object, issues: list[TaskDiscoveryIssue]) -> None:
        if not isinstance(value, Mapping):
            issues.append(TaskDiscoveryIssue("MISSING_LABEL_ACCESS", "label_access object is required"))
            return
        expected = {"train_targets": True, "validation_targets": True, "inference_targets": False}
        if any(value.get(key) is not expected_value for key, expected_value in expected.items()):
            issues.append(TaskDiscoveryIssue("LABEL_ACCESS_VIOLATION", "labels must be unavailable at inference"))

    @staticmethod
    def _validate_splits(value: object, issues: list[TaskDiscoveryIssue]) -> None:
        if not isinstance(value, Mapping):
            issues.append(TaskDiscoveryIssue("MISSING_SPLIT_DISCLOSURE", "splits object is required"))
            return
        if not isinstance(value.get("protocol"), str) or not value["protocol"].strip():
            issues.append(TaskDiscoveryIssue("MISSING_SPLIT_PROTOCOL", "splits.protocol is required"))
        if not isinstance(value.get("unit"), str) or not value["unit"].strip():
            issues.append(TaskDiscoveryIssue("MISSING_SPLIT_UNIT", "splits.unit is required"))

    @staticmethod
    def _validate_alignment(value: object, issues: list[TaskDiscoveryIssue]) -> None:
        if not isinstance(value, str) or not value.strip():
            issues.append(TaskDiscoveryIssue("MISSING_ALIGNMENT_SOURCE", "alignment_source is required"))

    def validate(self, source: str) -> TaskDiscoveryReport:
        return self.discover(source)

    def build_manifest(self, source: str) -> DatasetManifest:
        report = self.discover(source)
        descriptor = report.require_valid()
        label_access = descriptor["label_access"]
        return DatasetManifest(
            dataset_id=str(descriptor["dataset_id"]),
            samples=(),
            information_access=InformationAccess(
                train_target_text=bool(label_access["train_targets"]),
                validation_target_text=bool(label_access["validation_targets"]),
                inference_target_text=bool(label_access["inference_targets"]),
                split_definition=str(descriptor["splits"]["protocol"]),
                alignment_source=str(descriptor["alignment_source"]),
            ),
            description=f"{self.requirements.name} inventory only; no participant records parsed.",
            metadata={
                "inventory_only": True,
                "task": self.requirements.task,
                "modality": descriptor["modality"],
                "split_unit": descriptor["splits"]["unit"],
            },
        )

    def iter_samples(self, source: str) -> Iterator[NeuralTextSample]:
        self.discover(source).require_valid()
        return iter(())


class Brain2QwertyDiscoveryAdapter(TypedTaskDiscoveryAdapter):
    """Inventory BCBL typing layouts without loading FIF/EEG participant data."""

    requirements = TaskAdapterRequirements(
        name="brain2qwerty_discovery",
        task="typed_text",
        allowed_modalities=(Modality.MEG, Modality.EEG),
        required_paths=(),
    )


class T15DiscoveryAdapter(TypedTaskDiscoveryAdapter):
    """Inventory T15 copy-task layouts without loading neural tensors."""

    requirements = TaskAdapterRequirements(
        name="t15_discovery",
        task="copy_typing",
        allowed_modalities=(Modality.INTRACORTICAL,),
        required_paths=(
            ("t15_copyTaskData_description.csv", "file"),
            ("t15_copyTask_neuralData", "directory"),
        ),
    )
