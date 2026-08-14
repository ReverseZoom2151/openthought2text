"""A deterministic, non-participant fixture dataset for contract tests.

It intentionally represents constrained reading-like trials rather than any
claim about real neural decoding.  The adapter is useful for exercising the
manifest, registry, and split-audit path in a fresh checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator

from .audit import AuditReport, audit_splits
from .manifest import DatasetManifest, load_manifest, write_manifest
from .schema import (
    InformationAccess,
    Modality,
    NeuralTextSample,
    SignalReference,
    TextTarget,
    TimeInterval,
)


@dataclass(frozen=True, slots=True)
class SyntheticValidationReport:
    """Validation result returned by :class:`SyntheticNeuralTextAdapter`."""

    manifest: DatasetManifest
    split_audit: AuditReport
    missing_signal_files: tuple[Path, ...] = ()
    invalid_signal_files: tuple[Path, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.split_audit.passed
            and not self.missing_signal_files
            and not self.invalid_signal_files
        )


class SyntheticNeuralTextAdapter:
    """Generate a deterministic tiny EEG-like fixture with safe splits.

    ``discover`` is pure and can be called before artifacts exist.  ``generate``
    materializes an inspectable manifest and lightweight JSON signal fixtures;
    it is safe to call repeatedly with identical configuration.
    """

    name = "synthetic"
    dataset_id = "synthetic_neural_text_v1"
    _splits = ("train", "validation", "test")

    def __init__(
        self,
        *,
        subject_count: int = 3,
        trials_per_subject: int = 2,
        sample_rate_hz: float = 100.0,
        duration_s: float = 1.0,
        seed: int = 7,
    ) -> None:
        if subject_count < len(self._splits):
            raise ValueError(
                "subject_count must be at least three for train/validation/test splits"
            )
        if trials_per_subject < 1:
            raise ValueError("trials_per_subject must be positive")
        if sample_rate_hz <= 0 or duration_s <= 0:
            raise ValueError("sample_rate_hz and duration_s must be positive")
        self.subject_count = subject_count
        self.trials_per_subject = trials_per_subject
        self.sample_rate_hz = sample_rate_hz
        self.duration_s = duration_s
        self.seed = seed

    def _root(self, source: str) -> Path:
        if not source.strip():
            raise ValueError("source must be a non-empty path")
        return Path(source).expanduser()

    def _signal_values(self, subject_index: int) -> list[list[float]]:
        """Small deterministic two-channel sequence, expressed without NumPy."""
        steps = max(2, round(self.sample_rate_hz * self.duration_s))
        base = self.seed + subject_index * 17
        return [
            [round(((base + step * 3) % 23) / 23.0, 6) for step in range(steps)],
            [round(((base + step * 5 + 11) % 29) / 29.0, 6) for step in range(steps)],
        ]

    def _signal_path(self, root: Path, subject_index: int) -> Path:
        return root / "signals" / f"subject-{subject_index:02d}.json"

    def _signal_reference(self, root: Path, subject_index: int) -> SignalReference:
        content = json.dumps(self._signal_values(subject_index), separators=(",", ":"))
        return SignalReference(
            uri=str(self._signal_path(root, subject_index)),
            recording_id=f"synthetic-recording-{subject_index:02d}",
            sampling_rate_hz=self.sample_rate_hz,
            channel_count=2,
            channel_names=("SYN-A", "SYN-B"),
            checksum_sha256=sha256(content.encode("utf-8")).hexdigest(),
        )

    def discover(self, source: str) -> DatasetManifest:
        """Return the canonical deterministic manifest without reading signals."""
        root = self._root(source)
        samples: list[NeuralTextSample] = []
        for subject_index in range(self.subject_count):
            subject_id = f"synthetic-subject-{subject_index:02d}"
            split = self._splits[subject_index % len(self._splits)]
            signal = self._signal_reference(root, subject_index)
            for trial_index in range(self.trials_per_subject):
                start_s = trial_index * self.duration_s
                samples.append(
                    NeuralTextSample(
                        sample_id=f"synthetic-{subject_index:02d}-{trial_index:03d}",
                        dataset_id=self.dataset_id,
                        subject_id=subject_id,
                        modality=Modality.EEG,
                        signal=signal,
                        interval=TimeInterval(start_s, start_s + self.duration_s),
                        target=TextTarget(
                            "synthetic constrained utterance "
                            f"subject {subject_index:02d} trial {trial_index:03d}"
                        ),
                        split=split,
                        session_id="synthetic-session-01",
                        run_id="synthetic-run-01",
                        trial_id=f"trial-{trial_index:03d}",
                        group_ids=(f"subject:{subject_id}", f"recording:{signal.recording_id}"),
                        task="synthetic_constrained_reading",
                        metadata={"fixture": True, "seed": self.seed},
                    )
                )
        return DatasetManifest(
            dataset_id=self.dataset_id,
            samples=tuple(samples),
            information_access=InformationAccess(
                split_definition="subject_disjoint",
                alignment_source="synthetic_fixed_duration",
            ),
            source_url="synthetic://openthought2text/deterministic-fixture",
            license="Apache-2.0",
            description="Deterministic non-participant fixture for package tests.",
            metadata={
                "seed": self.seed,
                "subject_count": self.subject_count,
                "trials_per_subject": self.trials_per_subject,
            },
        )

    def build_manifest(self, source: str) -> DatasetManifest:
        """Protocol alias for :meth:`discover`."""
        return self.discover(source)

    def iter_samples(self, source: str) -> Iterator[NeuralTextSample]:
        yield from self.discover(source).samples

    def generate(self, source: str) -> DatasetManifest:
        """Materialize JSON signal fixtures and ``synthetic_manifest.jsonl``."""
        root = self._root(source)
        root.mkdir(parents=True, exist_ok=True)
        for subject_index in range(self.subject_count):
            signal_path = self._signal_path(root, subject_index)
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.write_text(
                json.dumps(self._signal_values(subject_index), separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        manifest = self.discover(str(root))
        write_manifest(root / "synthetic_manifest.jsonl", manifest)
        return manifest

    def validate(self, source: str) -> SyntheticValidationReport:
        """Load generated artifacts when present and audit their split contract."""
        root = self._root(source)
        manifest_path = root / "synthetic_manifest.jsonl"
        manifest = load_manifest(manifest_path) if manifest_path.exists() else self.discover(source)
        missing: set[Path] = set()
        invalid: set[Path] = set()
        for sample in manifest.samples:
            signal_path = Path(sample.signal.uri)
            if not signal_path.is_file():
                missing.add(signal_path)
                continue
            expected = sample.signal.checksum_sha256
            contents = signal_path.read_text(encoding="utf-8").strip()
            actual = sha256(contents.encode("utf-8")).hexdigest()
            if expected is not None and actual != expected:
                invalid.add(signal_path)
        return SyntheticValidationReport(
            manifest=manifest,
            split_audit=audit_splits(
                manifest.samples,
                information_access=manifest.information_access,
            ),
            missing_signal_files=tuple(sorted(missing)),
            invalid_signal_files=tuple(sorted(invalid)),
        )
