from __future__ import annotations

import inspect

from openthought2text.data import (
    Brain2QwertyWindowConfig,
    T15TargetAccessContract,
    plan_authorized_brain2qwerty_conversion,
    plan_authorized_t15_conversion,
    task_conversion_plans,
    validate_brain2qwerty_typed_events,
    validate_t15_descriptor_records,
)


class BrainReader:
    authorization_id = "ok"

    def read_typed_event_records(self, source):
        return [
            {
                "subject_id": "s1",
                "recording_id": "meg-r1",
                "modality": "meg",
                "sampling_rate_hz": 1000,
                "recording_duration_s": 2,
                "event": {"event_id": "e1", "typed_text": "a", "timestamp_s": 0.5},
            },
            {
                "subject_id": "s1",
                "recording_id": "eeg-r1",
                "modality": "eeg",
                "sampling_rate_hz": 1000,
                "recording_duration_s": 2,
                "event": {"event_id": "e2", "typed_text": "b", "timestamp_s": 1},
            },
        ]


class T15Reader:
    authorization_id = "ok"

    def read_descriptor_records(self, source):
        access = {"train_targets": True, "validation_targets": True, "inference_targets": False}
        return [
            {
                "subject_id": "t15",
                "block_id": "b1",
                "day_id": "d1",
                "session_id": "s1",
                "recording_id": "r1",
                "target_access": access,
            }
        ]


def test_authorized_brain2qwerty_plan_validates_modalities_windows_and_event_oracle_disclosure() -> (
    None
):
    config = Brain2QwertyWindowConfig(event_oracle_available=True)
    report, plan = plan_authorized_brain2qwerty_conversion(
        BrainReader(), authorization_id="ok", source_root_identifier="authorized_b2q", window=config
    )
    assert report.passed and plan is not None
    assert plan.to_dict()["window"]["event_oracle_available"] is True
    assert plan.event_count_by_modality == {"eeg": 1, "meg": 1}


def test_brain2qwerty_reports_mixed_modality_and_bad_window_without_raw_loading() -> None:
    records = list(BrainReader().read_typed_event_records("x"))
    records[1]["recording_id"] = "meg-r1"
    records[1]["event"]["timestamp_s"] = 1.8
    report = validate_brain2qwerty_typed_events(records, Brain2QwertyWindowConfig())
    assert {"MIXED_MODALITY_RECORDING", "WINDOW_EXCEEDS_RECORDING"} <= {
        item.code for item in report.errors
    }
    source = inspect.getsource(task_conversion_plans)
    assert "h5py" not in source and "loadmat" not in source


def test_t15_descriptor_plan_requires_block_day_session_mapping_and_target_contract() -> None:
    access = T15TargetAccessContract(True, True, False)
    report, plan = plan_authorized_t15_conversion(
        T15Reader(),
        authorization_id="ok",
        source_root_identifier="authorized_t15",
        target_access=access,
    )
    assert report.passed and plan is not None and plan.block_count == 1
    bad = [
        {
            "subject_id": "t15",
            "block_id": "b1",
            "day_id": "d1",
            "session_id": "s1",
            "recording_id": "r1",
            "target_access": {},
        }
    ]
    quality = validate_t15_descriptor_records(bad, access)
    assert "TARGET_ACCESS_MISMATCH" in {item.code for item in quality.errors}
