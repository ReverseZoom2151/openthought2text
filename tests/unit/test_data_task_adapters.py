from __future__ import annotations

import json

from openthought2text.data.task_adapters import (
    Brain2QwertyDiscoveryAdapter,
    T15DiscoveryAdapter,
)
from openthought2text.data.adapters import DatasetAdapter


def write_descriptor(root, *, dataset_id, modality, task, recordings_dir, split_manifest):
    descriptor = {
        "kind": "openthought2text.task_dataset_descriptor",
        "dataset_id": dataset_id,
        "modality": modality,
        "task": task,
        "recordings_dir": recordings_dir,
        "split_manifest": split_manifest,
        "alignment_source": "recorded_task_events",
        "label_access": {"train_targets": True, "validation_targets": True, "inference_targets": False},
        "splits": {"protocol": "subject_holdout", "unit": "subject"},
    }
    (root / "task_adapter.json").write_text(json.dumps(descriptor), encoding="utf-8")


def test_brain2qwerty_discovery_validates_typed_meg_inventory(tmp_path) -> None:
    (tmp_path / "MEG").mkdir()
    (tmp_path / "splits.json").write_text("{}", encoding="utf-8")
    write_descriptor(tmp_path, dataset_id="spanish_bcbl", modality="meg", task="typed_text", recordings_dir="MEG", split_manifest="splits.json")
    adapter = Brain2QwertyDiscoveryAdapter()
    report = adapter.discover(str(tmp_path))
    assert isinstance(adapter, DatasetAdapter)
    assert report.passed
    manifest = adapter.build_manifest(str(tmp_path))
    assert manifest.samples == ()
    assert manifest.information_access.inference_target_text is False


def test_t15_discovery_requires_copy_task_layout_and_intracortical_contract(tmp_path) -> None:
    (tmp_path / "t15_copyTask_neuralData").mkdir()
    (tmp_path / "t15_copyTaskData_description.csv").write_text("split\ntrain\n", encoding="utf-8")
    write_descriptor(tmp_path, dataset_id="t15_copy_task", modality="intracortical", task="copy_typing", recordings_dir="t15_copyTask_neuralData", split_manifest="t15_copyTaskData_description.csv")
    report = T15DiscoveryAdapter().validate(str(tmp_path))
    assert report.passed
    assert T15DiscoveryAdapter().build_manifest(str(tmp_path)).metadata["task"] == "copy_typing"


def test_discovery_reports_modality_label_access_and_required_path_violations(tmp_path) -> None:
    (tmp_path / "splits.json").write_text("{}", encoding="utf-8")
    write_descriptor(tmp_path, dataset_id="wrong", modality="intracortical", task="typed_text", recordings_dir="missing", split_manifest="splits.json")
    data = json.loads((tmp_path / "task_adapter.json").read_text(encoding="utf-8"))
    data["label_access"]["inference_targets"] = True
    (tmp_path / "task_adapter.json").write_text(json.dumps(data), encoding="utf-8")
    report = Brain2QwertyDiscoveryAdapter().discover(str(tmp_path))
    codes = {issue.code for issue in report.errors}
    assert {"MODALITY_MISMATCH", "LABEL_ACCESS_VIOLATION", "MISSING_DECLARED_PATH"} <= codes


def test_t15_discovery_reports_missing_fixed_paths(tmp_path) -> None:
    (tmp_path / "signals").mkdir()
    (tmp_path / "splits.csv").write_text("split\n", encoding="utf-8")
    write_descriptor(tmp_path, dataset_id="t15_copy_task", modality="intracortical", task="copy_typing", recordings_dir="signals", split_manifest="splits.csv")
    report = T15DiscoveryAdapter().discover(str(tmp_path))
    assert "MISSING_REQUIRED_PATH" in {issue.code for issue in report.errors}
