from __future__ import annotations

import inspect
from pathlib import Path

from openthought2text.data import (
    plan_authorized_zuco_raw_conversion,
    validate_zuco_alignment_records,
    zuco_raw_plan,
)


class FixtureAuthorizedReader:
    authorization_id = "fixture-authorization"

    def __init__(self, records):
        self.records = records
        self.calls = []

    def read_alignment_records(self, source_root_identifier):
        self.calls.append(source_root_identifier)
        return self.records


def valid_record():
    return {
        "subject_id": "ZAB",
        "task": "task1-SR",
        "sentence": {"sentence_id": "s-1", "text": "A short sentence."},
        "words": [
            {"word_index": 0, "text": "A", "start_s": 0.0, "end_s": 0.1},
            {"word_index": 1, "text": "short", "start_s": 0.1, "end_s": 0.3},
        ],
        "fixations": [{"word_index": 1, "start_s": 0.12, "end_s": 0.2}],
        "eeg": {"recording_id": "ZAB-task1", "sampling_rate_hz": 500, "channel_count": 105},
    }


def test_authorized_reader_produces_text_minimized_conversion_plan_without_matlab_loader() -> None:
    reader = FixtureAuthorizedReader([valid_record()])
    report, plan = plan_authorized_zuco_raw_conversion(
        reader,
        authorization_id="fixture-authorization",
        source_root_identifier="authorized_zuco_fixture",
    )
    assert report.passed and plan is not None
    assert reader.calls == ["authorized_zuco_fixture"]
    assert plan.records[0].sentence_text_sha256 != valid_record()["sentence"]["text"]
    assert "loadmat" not in inspect.getsource(zuco_raw_plan)
    assert "scipy" not in inspect.getsource(zuco_raw_plan)
    repository = Path(__file__).resolve().parents[2]
    assert not list(repository.rglob("*.mat"))


def test_alignment_quality_report_is_actionable_for_partial_and_malformed_records() -> None:
    bad = valid_record()
    del bad["fixations"]
    bad["words"][1]["word_index"] = 3
    bad["eeg"]["channel_count"] = 0
    report, records = validate_zuco_alignment_records([bad, {"subject_id": "partial"}])
    assert not report.passed
    assert records == ()
    codes = {issue.code for issue in report.errors}
    assert {
        "MISSING_FIXATION_ALIGNMENT",
        "NONCANONICAL_WORD_INDICES",
        "MALFORMED_EEG_ALIGNMENT",
        "MISSING_SENTENCE_ALIGNMENT",
    } <= codes
