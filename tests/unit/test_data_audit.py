from openthought2text.data import InformationAccess, PretrainingExposure, audit_splits

from .test_data_schema import sample


def codes(report):
    return {finding.code for finding in report.findings}


def test_audit_detects_text_group_and_interval_leakage():
    train = sample(sample_id="train", group_ids=("story:1",), split="train")
    test = sample(sample_id="test", split="test", group_ids=("story:1",), interval=train.interval)
    report = audit_splits((train, test))
    assert {"DUPLICATE_TARGET_TEXT", "GROUP_ACROSS_SPLITS", "CONTINUOUS_INTERVAL_OVERLAP"} <= codes(
        report
    )
    assert not report.passed


def test_audit_detects_declared_pretraining_overlap_and_inference_text():
    row = sample()
    report = audit_splits(
        (row,),
        information_access=InformationAccess(inference_target_text=True),
        pretraining=PretrainingExposure(sample_ids=frozenset({row.sample_id}), declared=True),
    )
    assert {"INFERENCE_TEXT_ACCESS", "PRETRAINING_OVERLAP"} <= codes(report)


def test_undeclared_pretraining_is_a_warning_not_a_clean_claim():
    report = audit_splits((sample(),), pretraining=PretrainingExposure())
    assert report.passed
    assert "PRETRAINING_PROVENANCE_UNDECLARED" in codes(report)


def test_audit_labels_aligned_access_and_rejects_audio_access():
    report = audit_splits(
        (sample(),),
        information_access=InformationAccess(
            inference_token_boundaries=True,
            inference_event_boundaries=True,
            inference_stimulus_audio=True,
        ),
    )
    assert {
        "INFERENCE_TOKEN_BOUNDARIES",
        "INFERENCE_EVENT_BOUNDARIES",
        "INFERENCE_STIMULUS_AUDIO",
    } <= codes(report)
    assert not report.passed
