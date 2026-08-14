"""Dependency-light ZuCo raw-layout discovery and validation.

This module does not open MATLAB files or infer neural/text examples.  Its
purpose is to make an installation's raw layout explicit before a later,
versioned converter reads participant data with an approved MATLAB backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Iterator

from .manifest import DatasetManifest
from .schema import InformationAccess, NeuralTextSample


_TASK_PATTERN = re.compile(r"^task(?P<number>[123])-(?P<code>[A-Za-z]+)(?P<version>-2\.0)?$")
_SUBJECT_PATTERN = re.compile(r"^results(?P<subject>[A-Za-z0-9]+)(?:[_-].*)?\.mat$", re.IGNORECASE)


class ZuCoLayoutSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ZuCoLayoutIssue:
    code: str
    severity: ZuCoLayoutSeverity
    message: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class ZuCoTaskInventory:
    """One discovered ZuCo task directory, without inspecting MATLAB content."""

    task_name: str
    task_path: Path
    matlab_files_path: Path | None
    mat_files: tuple[Path, ...] = ()
    subject_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ZuCoDiscoveryReport:
    root: Path
    tasks: tuple[ZuCoTaskInventory, ...] = ()
    issues: tuple[ZuCoLayoutIssue, ...] = ()

    @property
    def errors(self) -> tuple[ZuCoLayoutIssue, ...]:
        return tuple(item for item in self.issues if item.severity == ZuCoLayoutSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ZuCoLayoutIssue, ...]:
        return tuple(item for item in self.issues if item.severity == ZuCoLayoutSeverity.WARNING)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def missing_fields(self) -> tuple[ZuCoLayoutIssue, ...]:
        return tuple(item for item in self.issues if item.code.startswith("MISSING_"))

    @property
    def ambiguous_fields(self) -> tuple[ZuCoLayoutIssue, ...]:
        return tuple(item for item in self.issues if item.code.startswith("AMBIGUOUS_"))

    def require_valid_layout(self) -> None:
        if self.errors:
            codes = ", ".join(issue.code for issue in self.errors)
            raise ValueError(f"ZuCo raw-layout validation failed: {codes}")


class ZuCoDiscoveryAdapter:
    """Inventory ZuCo v1/v2-style directory layouts without MATLAB dependencies.

    Recognized task folders have the form ``task1-SR``, ``task2-NR``, or
    ``task3-TSR``, optionally with the ``-2.0`` suffix.  Each is expected to
    have a ``Matlab_files`` directory containing files named like
    ``resultsZAB.mat`` or ``resultsZAB_NR.mat``.
    """

    name = "zuco_discovery"
    dataset_id = "zuco_raw_layout"

    def _root(self, source: str) -> Path:
        if not source.strip():
            raise ValueError("source must be a non-empty path")
        return Path(source).expanduser()

    @staticmethod
    def _task_directories(root: Path) -> tuple[tuple[str, Path], ...]:
        if _TASK_PATTERN.match(root.name):
            return ((root.name, root),)
        return tuple(
            (child.name, child)
            for child in sorted(root.iterdir())
            if child.is_dir() and _TASK_PATTERN.match(child.name)
        )

    def discover(self, source: str) -> ZuCoDiscoveryReport:
        """Inspect a directory tree only; never import or execute MATLAB tooling."""
        root = self._root(source)
        if not root.is_dir():
            return ZuCoDiscoveryReport(
                root=root,
                issues=(
                    ZuCoLayoutIssue(
                        "MISSING_SOURCE_DIRECTORY",
                        ZuCoLayoutSeverity.ERROR,
                        "source directory does not exist or is not a directory",
                        root,
                    ),
                ),
            )

        issues: list[ZuCoLayoutIssue] = []
        task_directories = self._task_directories(root)
        if not task_directories:
            issues.append(
                ZuCoLayoutIssue(
                    "MISSING_TASK_DIRECTORY",
                    ZuCoLayoutSeverity.ERROR,
                    "no ZuCo-style task directory was found under source",
                    root,
                )
            )
            return ZuCoDiscoveryReport(root=root, issues=tuple(issues))

        logical_tasks: dict[tuple[str, str], list[Path]] = {}
        inventories: list[ZuCoTaskInventory] = []
        for task_name, task_path in task_directories:
            match = _TASK_PATTERN.match(task_name)
            assert match is not None  # Filtered by _task_directories.
            key = (match.group("number"), match.group("code").casefold())
            logical_tasks.setdefault(key, []).append(task_path)
            matlab_path = task_path / "Matlab_files"
            if not matlab_path.is_dir():
                issues.append(
                    ZuCoLayoutIssue(
                        "MISSING_MATLAB_FILES_DIRECTORY",
                        ZuCoLayoutSeverity.ERROR,
                        "task is missing its expected Matlab_files directory",
                        matlab_path,
                    )
                )
                inventories.append(ZuCoTaskInventory(task_name, task_path, None))
                continue

            mat_files = tuple(
                sorted(
                    path
                    for path in matlab_path.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".mat"
                )
            )
            if not mat_files:
                issues.append(
                    ZuCoLayoutIssue(
                        "MISSING_MAT_FILES",
                        ZuCoLayoutSeverity.ERROR,
                        "Matlab_files contains no .mat recording files",
                        matlab_path,
                    )
                )

            subjects: list[str] = []
            for mat_file in mat_files:
                subject_match = _SUBJECT_PATTERN.match(mat_file.name)
                if subject_match is None:
                    issues.append(
                        ZuCoLayoutIssue(
                            "AMBIGUOUS_SUBJECT_ID",
                            ZuCoLayoutSeverity.ERROR,
                            "cannot derive a subject identifier from MATLAB filename",
                            mat_file,
                        )
                    )
                    continue
                subjects.append(subject_match.group("subject").upper())
            duplicates = sorted({subject for subject in subjects if subjects.count(subject) > 1})
            for subject in duplicates:
                issues.append(
                    ZuCoLayoutIssue(
                        "AMBIGUOUS_SUBJECT_FILES",
                        ZuCoLayoutSeverity.ERROR,
                        f"more than one MATLAB file maps to subject {subject!r} in this task",
                        matlab_path,
                    )
                )
            inventories.append(
                ZuCoTaskInventory(
                    task_name=task_name,
                    task_path=task_path,
                    matlab_files_path=matlab_path,
                    mat_files=mat_files,
                    subject_ids=tuple(sorted(set(subjects))),
                )
            )

        for (_, _), paths in logical_tasks.items():
            if len(paths) > 1:
                issues.append(
                    ZuCoLayoutIssue(
                        "AMBIGUOUS_TASK_VERSION",
                        ZuCoLayoutSeverity.ERROR,
                        "multiple versioned directories represent the same logical task",
                        root,
                    )
                )
        expected = {"task1-SR", "task2-NR", "task3-TSR"}
        observed = {task.task_name.removesuffix("-2.0") for task in inventories}
        if observed != expected:
            missing = ", ".join(sorted(expected - observed)) or "none"
            issues.append(
                ZuCoLayoutIssue(
                    "PARTIAL_TASK_LAYOUT",
                    ZuCoLayoutSeverity.WARNING,
                    f"not all ZuCo v1 task folders are present; missing: {missing}",
                    root,
                )
            )
        return ZuCoDiscoveryReport(root, tuple(inventories), tuple(issues))

    def validate(self, source: str) -> ZuCoDiscoveryReport:
        """Alias that makes validation intent explicit to callers."""
        return self.discover(source)

    def build_manifest(self, source: str) -> DatasetManifest:
        """Build an inventory-only manifest after raw-layout validation.

        This never claims to have parsed participant signals.  A later raw-data
        converter must create the non-empty canonical sample manifest.
        """
        report = self.discover(source)
        report.require_valid_layout()
        return DatasetManifest(
            dataset_id=self.dataset_id,
            samples=(),
            information_access=InformationAccess(
                split_definition="unassigned_pending_raw_conversion",
                alignment_source="uninspected_matlab_payload",
            ),
            description="ZuCo raw-layout inventory only; no participant samples parsed.",
            metadata={
                "inventory_only": True,
                "source_root": str(report.root),
                "tasks": [
                    {
                        "name": task.task_name,
                        "mat_file_count": len(task.mat_files),
                        "subject_ids": list(task.subject_ids),
                    }
                    for task in report.tasks
                ],
            },
        )

    def iter_samples(self, source: str) -> Iterator[NeuralTextSample]:
        """Yield no samples: this adapter deliberately does not parse MATLAB payloads."""
        self.discover(source).require_valid_layout()
        return iter(())
