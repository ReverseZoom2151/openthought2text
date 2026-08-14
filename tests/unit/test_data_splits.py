from __future__ import annotations

from dataclasses import replace

from openthought2text.data import (
    SplitPlan,
    SplitProtocol,
    build_split_plan,
    validate_split_plan,
)

from .test_data_schema import sample


def rows() -> tuple:
    result = []
    for index, (subject, session, text) in enumerate(
        (
            ("a", "session-a", "alpha"),
            ("a", "session-a", "beta"),
            ("b", "session-b", "alpha"),
            ("b", "session-b", "gamma"),
            ("c", "session-c", "delta"),
            ("c", "session-c", "epsilon"),
        )
    ):
        row = sample(
            sample_id=f"s-{index}",
            subject_id=subject,
            session_id=session,
            interval=sample().interval.__class__(index * 2.0, index * 2.0 + 1.0),
            target=sample().target.__class__(text),
            group_ids=(),
        )
        result.append(row)
    return tuple(result)


def test_random_legacy_is_deterministic_and_manifest_ready() -> None:
    source = rows()
    first = build_split_plan(source, "random_legacy", seed=13)
    second = build_split_plan(source, SplitProtocol.RANDOM_LEGACY, seed=13)

    assert first == second
    assert {sample.split for sample in first.materialize(source)} <= {"train", "validation", "test"}
    assert validate_split_plan(source, first).passed


def test_unique_text_keeps_repeated_targets_together() -> None:
    source = rows()
    plan = build_split_plan(source, "unique_text", seed=5)
    assignments = plan.assignment_map

    assert assignments["s-0"] == assignments["s-2"]
    assert validate_split_plan(source, plan).passed


def test_session_holdout_keeps_each_session_together() -> None:
    source = rows()
    plan = build_split_plan(source, "session_holdout", seed=3)
    assignments = plan.assignment_map

    assert assignments["s-0"] == assignments["s-1"]
    assert assignments["s-2"] == assignments["s-3"]
    assert validate_split_plan(source, plan).passed


def test_loso_subject_places_only_selected_subject_in_test() -> None:
    source = rows()
    plan = build_split_plan(source, "loso_subject", held_out_subject="c", seed=2)
    materialized = plan.materialize(source)

    assert plan.held_out_subject == "c"
    assert {sample.subject_id for sample in materialized if sample.split == "test"} == {"c"}
    assert validate_split_plan(source, plan).passed


def test_loso_subject_unique_text_excludes_nonheldout_target_collision() -> None:
    source = rows()
    plan = build_split_plan(source, "loso_subject_unique_text", held_out_subject="b", seed=2)

    assert "s-0" in plan.excluded_sample_ids  # alpha also occurs in held-out subject b.
    assert all(sample.sample_id != "s-0" for sample in plan.materialize(source))
    assert validate_split_plan(source, plan).passed


def test_validator_rejects_forbidden_unique_text_overlap() -> None:
    source = rows()
    plan = SplitPlan(
        protocol=SplitProtocol.UNIQUE_TEXT,
        seed=0,
        assignments=tuple(
            (row.sample_id, "train" if row.sample_id == "s-0" else "test")
            for row in source
        ),
    )
    report = validate_split_plan(source, plan)

    assert "TEXT_ACROSS_SPLITS" in {item.code for item in report.violations}
