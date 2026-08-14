"""Leakage checks for declared neural-text splits and pretraining exposure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from .schema import InformationAccess, NeuralTextSample


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: Severity
    message: str
    sample_ids: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PretrainingExposure:
    """Known source identities used for representation/pretraining runs.

    Callers can populate as much as is available.  Empty sets mean unknown,
    rather than an assertion that no overlap exists.
    """

    dataset_ids: frozenset[str] = frozenset()
    recording_keys: frozenset[tuple[str, str]] = frozenset()
    sample_ids: frozenset[str] = frozenset()
    target_fingerprints: frozenset[str] = frozenset()
    declared: bool = False


@dataclass(frozen=True, slots=True)
class AuditReport:
    findings: tuple[AuditFinding, ...] = ()
    sample_count: int = 0

    @property
    def errors(self) -> tuple[AuditFinding, ...]:
        return tuple(item for item in self.findings if item.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[AuditFinding, ...]:
        return tuple(item for item in self.findings if item.severity == Severity.WARNING)

    @property
    def passed(self) -> bool:
        return not self.errors

    def require_clean(self) -> None:
        if self.errors:
            codes = ", ".join(item.code for item in self.errors)
            raise ValueError(f"split audit failed: {codes}")


def _distinct_splits(samples: Iterable[NeuralTextSample]) -> tuple[str, ...]:
    return tuple(sorted({sample.split for sample in samples if sample.split is not None}))


def audit_splits(
    samples: Iterable[NeuralTextSample],
    *,
    information_access: InformationAccess | None = None,
    pretraining: PretrainingExposure | None = None,
    interval_tolerance_s: float = 0.0,
    reject_duplicate_text: bool = True,
) -> AuditReport:
    """Audit all cross-split information paths without reading signal arrays.

    It reports missing split labels, split-group reuse, exact normalized target
    reuse, overlapping continuous windows from the same recording, declared
    pretraining overlap, and target visibility at inference.
    """
    if interval_tolerance_s < 0:
        raise ValueError("interval_tolerance_s must be non-negative")
    rows = tuple(samples)
    findings: list[AuditFinding] = []
    seen_ids: set[str] = set()
    for sample in rows:
        if sample.sample_id in seen_ids:
            findings.append(AuditFinding("DUPLICATE_SAMPLE_ID", Severity.ERROR,
                                         "sample_id appears more than once", (sample.sample_id,)))
        seen_ids.add(sample.sample_id)
        if sample.split is None:
            findings.append(AuditFinding("MISSING_SPLIT", Severity.ERROR,
                                         "sample lacks a declared split", (sample.sample_id,)))

    if information_access and information_access.inference_label_leakage:
        findings.append(AuditFinding(
            "INFERENCE_TEXT_ACCESS", Severity.ERROR,
            "target text or text context is declared visible at inference; free generation is invalid",
        ))
    if information_access and information_access.inference_token_boundaries:
        findings.append(AuditFinding(
            "INFERENCE_TOKEN_BOUNDARIES", Severity.WARNING,
            "gold token boundaries are visible at inference; report this as an aligned regime",
        ))
    if information_access and information_access.inference_event_boundaries:
        findings.append(AuditFinding(
            "INFERENCE_EVENT_BOUNDARIES", Severity.WARNING,
            "event boundaries are visible at inference; report this as an event-aligned regime",
        ))
    if information_access and information_access.inference_stimulus_audio:
        findings.append(AuditFinding(
            "INFERENCE_STIMULUS_AUDIO", Severity.ERROR,
            "stimulus audio is declared visible at inference; this is not neural-only decoding",
        ))

    by_group: dict[str, list[NeuralTextSample]] = defaultdict(list)
    by_text: dict[str, list[NeuralTextSample]] = defaultdict(list)
    by_recording: dict[tuple[str, str], list[NeuralTextSample]] = defaultdict(list)
    for sample in rows:
        for group_id in sample.group_ids:
            by_group[group_id].append(sample)
        if sample.target:
            by_text[sample.target.fingerprint].append(sample)
        by_recording[sample.recording_key].append(sample)

    for group_id, grouped in sorted(by_group.items()):
        splits = _distinct_splits(grouped)
        if len(splits) > 1:
            findings.append(AuditFinding(
                "GROUP_ACROSS_SPLITS", Severity.ERROR,
                f"group {group_id!r} occurs in multiple splits",
                tuple(sorted(sample.sample_id for sample in grouped)), splits,
            ))

    for fingerprint, grouped in sorted(by_text.items()):
        splits = _distinct_splits(grouped)
        if len(splits) > 1:
            severity = Severity.ERROR if reject_duplicate_text else Severity.WARNING
            findings.append(AuditFinding(
                "DUPLICATE_TARGET_TEXT", severity,
                f"normalized target text {fingerprint[:12]}… occurs in multiple splits",
                tuple(sorted(sample.sample_id for sample in grouped)), splits,
            ))

    for recording_key, grouped in sorted(by_recording.items()):
        ordered = sorted(grouped, key=lambda sample: sample.interval.start_s)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right.interval.start_s >= left.interval.end_s + interval_tolerance_s:
                    break
                if left.split != right.split and left.split is not None and right.split is not None:
                    findings.append(AuditFinding(
                        "CONTINUOUS_INTERVAL_OVERLAP", Severity.ERROR,
                        f"overlapping windows in recording {recording_key!r} cross splits",
                        tuple(sorted((left.sample_id, right.sample_id))),
                        tuple(sorted((left.split, right.split))),
                    ))

    if pretraining is not None:
        if not pretraining.declared:
            findings.append(AuditFinding(
                "PRETRAINING_PROVENANCE_UNDECLARED", Severity.WARNING,
                "no declared pretraining exposure; overlap cannot be ruled out",
            ))
        for sample in rows:
            overlap: str | None = None
            if sample.sample_id in pretraining.sample_ids:
                overlap = "sample"
            elif sample.recording_key in pretraining.recording_keys:
                overlap = "recording"
            elif sample.dataset_id in pretraining.dataset_ids:
                overlap = "dataset"
            elif sample.target and sample.target.fingerprint in pretraining.target_fingerprints:
                overlap = "target text"
            if overlap:
                findings.append(AuditFinding(
                    "PRETRAINING_OVERLAP", Severity.ERROR,
                    f"{overlap} was declared as used for pretraining",
                    (sample.sample_id,), (sample.split,) if sample.split else (),
                ))

    return AuditReport(tuple(findings), sample_count=len(rows))
