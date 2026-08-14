"""Deterministic, leakage-aware split construction over canonical samples."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import random
from typing import Iterable

from .schema import NeuralTextSample


class SplitProtocol(str, Enum):
    RANDOM_LEGACY = "random_legacy"
    UNIQUE_TEXT = "unique_text"
    SESSION_HOLDOUT = "session_holdout"
    LOSO_SUBJECT = "loso_subject"
    LOSO_SUBJECT_UNIQUE_TEXT = "loso_subject_unique_text"


@dataclass(frozen=True, slots=True)
class SplitViolation:
    code: str
    message: str
    sample_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SplitValidationReport:
    protocol: SplitProtocol
    violations: tuple[SplitViolation, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations

    def require_valid(self) -> None:
        if self.violations:
            codes = ", ".join(item.code for item in self.violations)
            raise ValueError(f"split plan violates protocol: {codes}")


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Manifest-ready assignments, with exclusions explicit rather than hidden."""

    protocol: SplitProtocol
    seed: int
    assignments: tuple[tuple[str, str], ...]
    excluded_sample_ids: tuple[str, ...] = ()
    held_out_subject: str | None = None

    def __post_init__(self) -> None:
        ids = [sample_id for sample_id, _ in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("split plan assignment sample IDs must be unique")
        if set(ids) & set(self.excluded_sample_ids):
            raise ValueError("a sample cannot be assigned and excluded")
        if any(split not in {"train", "validation", "test"} for _, split in self.assignments):
            raise ValueError("split assignments must use train, validation, or test")

    @property
    def assignment_map(self) -> dict[str, str]:
        return dict(self.assignments)

    def materialize(self, samples: Iterable[NeuralTextSample]) -> tuple[NeuralTextSample, ...]:
        """Return samples in source order with the assigned split field replaced."""
        rows = tuple(samples)
        assignment_map = self.assignment_map
        known = {sample.sample_id for sample in rows}
        expected = set(assignment_map) | set(self.excluded_sample_ids)
        if known != expected:
            missing = sorted(known - expected)
            unknown = sorted(expected - known)
            raise ValueError(f"split plan/sample IDs differ; missing={missing}, unknown={unknown}")
        excluded = set(self.excluded_sample_ids)
        return tuple(
            replace(sample, split=assignment_map[sample.sample_id])
            for sample in rows
            if sample.sample_id not in excluded
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol.value,
            "seed": self.seed,
            "assignments": [
                {"sample_id": sample_id, "split": split}
                for sample_id, split in self.assignments
            ],
            "excluded_sample_ids": list(self.excluded_sample_ids),
            "held_out_subject": self.held_out_subject,
        }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root

    def groups(self) -> tuple[tuple[int, ...], ...]:
        result: dict[int, list[int]] = {}
        for index in range(len(self.parents)):
            result.setdefault(self.find(index), []).append(index)
        return tuple(tuple(indices) for _, indices in sorted(result.items()))


def _require_unique_ids(rows: tuple[NeuralTextSample, ...]) -> None:
    ids = [sample.sample_id for sample in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("split construction requires a non-empty collection of unique sample IDs")


def _require_targets(rows: tuple[NeuralTextSample, ...]) -> None:
    missing = [sample.sample_id for sample in rows if sample.target is None]
    if missing:
        raise ValueError("text-aware split requires targets for: " + ", ".join(sorted(missing)))


def _overlap_components(rows: tuple[NeuralTextSample, ...]) -> _DisjointSet:
    """Unite all overlapping windows from a single physical recording."""
    groups = _DisjointSet(len(rows))
    by_recording: dict[tuple[str, str], list[int]] = {}
    for index, sample in enumerate(rows):
        by_recording.setdefault(sample.recording_key, []).append(index)
    for indices in by_recording.values():
        ordered = sorted(indices, key=lambda index: rows[index].interval.start_s)
        active: list[int] = []
        for index in ordered:
            active = [
                other
                for other in active
                if rows[other].interval.end_s > rows[index].interval.start_s
            ]
            for other in active:
                groups.union(index, other)
            active.append(index)
    return groups


def _units(rows: tuple[NeuralTextSample, ...], *, mode: str) -> tuple[tuple[int, ...], ...]:
    groups = _overlap_components(rows)
    index_by_text: dict[str, int] = {}
    index_by_session: dict[str, int] = {}
    for index, sample in enumerate(rows):
        if mode == "text":
            assert sample.target is not None
            fingerprint = sample.target.fingerprint
            if fingerprint in index_by_text:
                groups.union(index, index_by_text[fingerprint])
            index_by_text[fingerprint] = index
        if mode == "session":
            if sample.session_id is None:
                raise ValueError("session_holdout requires every sample to have session_id")
            if sample.session_id in index_by_session:
                groups.union(index, index_by_session[sample.session_id])
            index_by_session[sample.session_id] = index
    return groups.groups()


def _assign_units(
    units: tuple[tuple[int, ...], ...],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[int, str]:
    if not 0 <= validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1) and test_fraction in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")
    if len(units) < 2:
        raise ValueError("at least two independent split units are required")
    ordered = sorted(units, key=lambda unit: tuple(unit))
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    test_count = max(1, round(count * test_fraction))
    validation_count = 0
    if validation_fraction > 0 and count >= 3:
        validation_count = max(1, round(count * validation_fraction))
    while test_count + validation_count >= count:
        if validation_count:
            validation_count -= 1
        else:
            test_count -= 1
    assigned: dict[int, str] = {}
    for unit in ordered[:test_count]:
        assigned.update({index: "test" for index in unit})
    for unit in ordered[test_count : test_count + validation_count]:
        assigned.update({index: "validation" for index in unit})
    for unit in ordered[test_count + validation_count :]:
        assigned.update({index: "train" for index in unit})
    return assigned


def _assign_non_test_units(
    units: tuple[tuple[int, ...], ...], *, seed: int, validation_fraction: float
) -> dict[int, str]:
    """Assign remaining LOSO units to train/validation; test is held-out only."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    ordered = sorted(units, key=lambda unit: tuple(unit))
    random.Random(seed).shuffle(ordered)
    validation_count = 0
    if validation_fraction > 0 and len(ordered) >= 2:
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_count = min(validation_count, len(ordered) - 1)
    assigned: dict[int, str] = {}
    for unit in ordered[:validation_count]:
        assigned.update({index: "validation" for index in unit})
    for unit in ordered[validation_count:]:
        assigned.update({index: "train" for index in unit})
    return assigned


def build_split_plan(
    samples: Iterable[NeuralTextSample],
    protocol: SplitProtocol | str,
    *,
    seed: int = 0,
    held_out_subject: str | None = None,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> SplitPlan:
    """Build a deterministic plan and reject any protocol-violating result."""
    rows = tuple(samples)
    _require_unique_ids(rows)
    resolved_protocol = SplitProtocol(protocol)
    excluded_indices: set[int] = set()
    held_out: str | None = None

    if resolved_protocol in {SplitProtocol.UNIQUE_TEXT, SplitProtocol.LOSO_SUBJECT_UNIQUE_TEXT}:
        _require_targets(rows)
    if resolved_protocol == SplitProtocol.RANDOM_LEGACY:
        assigned = _assign_units(
            _units(rows, mode="none"),
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    elif resolved_protocol == SplitProtocol.UNIQUE_TEXT:
        assigned = _assign_units(
            _units(rows, mode="text"),
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    elif resolved_protocol == SplitProtocol.SESSION_HOLDOUT:
        assigned = _assign_units(
            _units(rows, mode="session"),
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    else:
        subject_ids = tuple(sorted({sample.subject_id for sample in rows}))
        if len(subject_ids) < 2:
            raise ValueError("LOSO requires at least two distinct subjects")
        held_out = held_out_subject or random.Random(seed).choice(subject_ids)
        if held_out not in subject_ids:
            raise ValueError(f"held_out_subject is absent from samples: {held_out!r}")
        held_indices = {index for index, sample in enumerate(rows) if sample.subject_id == held_out}
        if resolved_protocol == SplitProtocol.LOSO_SUBJECT_UNIQUE_TEXT:
            assert all(rows[index].target is not None for index in held_indices)
            held_texts = {
                rows[index].target.fingerprint
                for index in held_indices
                if rows[index].target
            }
            excluded_indices = {
                index
                for index, sample in enumerate(rows)
                if (
                    sample.subject_id != held_out
                    and sample.target
                    and sample.target.fingerprint in held_texts
                )
            }
        candidate_indices = [
            index for index, sample in enumerate(rows)
            if sample.subject_id != held_out and index not in excluded_indices
        ]
        if not candidate_indices:
            raise ValueError("LOSO leaves no non-held-out samples after exclusion")
        candidates = tuple(rows[index] for index in candidate_indices)
        mode = "text" if resolved_protocol == SplitProtocol.LOSO_SUBJECT_UNIQUE_TEXT else "none"
        candidate_assignments = _assign_non_test_units(
            _units(candidates, mode=mode),
            seed=seed,
            validation_fraction=validation_fraction,
        )
        assigned = {index: "test" for index in held_indices}
        assigned.update({
            candidate_indices[candidate_index]: split
            for candidate_index, split in candidate_assignments.items()
        })

    plan = SplitPlan(
        protocol=resolved_protocol,
        seed=seed,
        assignments=tuple(
            sorted((rows[index].sample_id, split) for index, split in assigned.items())
        ),
        excluded_sample_ids=tuple(sorted(rows[index].sample_id for index in excluded_indices)),
        held_out_subject=held_out,
    )
    validate_split_plan(rows, plan).require_valid()
    return plan


def validate_split_plan(
    samples: Iterable[NeuralTextSample], plan: SplitPlan
) -> SplitValidationReport:
    """Validate assignment completeness plus the no-overlap rule of each protocol."""
    rows = tuple(samples)
    violations: list[SplitViolation] = []
    ids = [sample.sample_id for sample in rows]
    if len(ids) != len(set(ids)):
        violations.append(
            SplitViolation("DUPLICATE_SAMPLE_ID", "source samples have duplicate sample IDs")
        )
        return SplitValidationReport(plan.protocol, tuple(violations))
    assignments = plan.assignment_map
    known = set(ids)
    expected = set(assignments) | set(plan.excluded_sample_ids)
    if known != expected:
        violations.append(
            SplitViolation(
                "INCOMPLETE_ASSIGNMENT",
                "plan does not cover exactly the source samples",
            )
        )
        return SplitValidationReport(plan.protocol, tuple(violations))

    active = [
        (sample, assignments[sample.sample_id])
        for sample in rows
        if sample.sample_id in assignments
    ]
    by_recording: dict[tuple[str, str], list[tuple[NeuralTextSample, str]]] = {}
    for sample, split in active:
        by_recording.setdefault(sample.recording_key, []).append((sample, split))
    for recording_rows in by_recording.values():
        for left_index, (left, left_split) in enumerate(recording_rows):
            for right, right_split in recording_rows[left_index + 1 :]:
                if left_split != right_split and left.interval.overlaps(right.interval):
                    violations.append(
                        SplitViolation(
                            "CONTINUOUS_INTERVAL_OVERLAP",
                            "overlapping recording windows cross split boundaries",
                            tuple(sorted((left.sample_id, right.sample_id))),
                        )
                    )

    if plan.protocol in {SplitProtocol.UNIQUE_TEXT, SplitProtocol.LOSO_SUBJECT_UNIQUE_TEXT}:
        by_text: dict[str, list[tuple[NeuralTextSample, str]]] = {}
        for sample, split in active:
            if sample.target is None:
                violations.append(
                    SplitViolation(
                        "MISSING_TARGET",
                        "text-aware protocol needs every target",
                        (sample.sample_id,),
                    )
                )
                continue
            by_text.setdefault(sample.target.fingerprint, []).append((sample, split))
        for text_rows in by_text.values():
            if len({split for _, split in text_rows}) > 1:
                violations.append(
                    SplitViolation(
                        "TEXT_ACROSS_SPLITS",
                        "the same normalized target text crosses splits",
                        tuple(sorted(sample.sample_id for sample, _ in text_rows)),
                    )
                )

    if plan.protocol == SplitProtocol.SESSION_HOLDOUT:
        by_session: dict[str, list[tuple[NeuralTextSample, str]]] = {}
        for sample, split in active:
            if sample.session_id is None:
                violations.append(
                    SplitViolation(
                        "MISSING_SESSION",
                        "session_holdout requires session_id",
                        (sample.sample_id,),
                    )
                )
                continue
            by_session.setdefault(sample.session_id, []).append((sample, split))
        for session_rows in by_session.values():
            if len({split for _, split in session_rows}) > 1:
                violations.append(
                    SplitViolation(
                        "SESSION_ACROSS_SPLITS",
                        "one session crosses split boundaries",
                        tuple(sorted(sample.sample_id for sample, _ in session_rows)),
                    )
                )

    if plan.protocol in {SplitProtocol.LOSO_SUBJECT, SplitProtocol.LOSO_SUBJECT_UNIQUE_TEXT}:
        if plan.held_out_subject is None:
            violations.append(
                SplitViolation("MISSING_HELD_OUT_SUBJECT", "LOSO plan needs held_out_subject")
            )
        else:
            for sample, split in active:
                if sample.subject_id == plan.held_out_subject and split != "test":
                    violations.append(
                        SplitViolation(
                            "HELD_OUT_SUBJECT_NOT_TEST",
                            "held-out subject is not entirely test",
                        )
                    )
                    break
                if sample.subject_id != plan.held_out_subject and split == "test":
                    violations.append(
                        SplitViolation(
                            "NON_HELD_OUT_SUBJECT_IN_TEST",
                            "test split includes another subject",
                        )
                    )
                    break
    return SplitValidationReport(plan.protocol, tuple(violations))
