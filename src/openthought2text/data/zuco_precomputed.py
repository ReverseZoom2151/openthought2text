"""Safe discovery contract for authorized precomputed ZuCo feature artifacts.

The adapter validates the canonical JSONL manifest and feature-file bytes, but
never calls ``torch.load`` and never reads raw MATLAB participant recordings.
This keeps discovery safe for portable, restricted research artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterator

from .manifest import DatasetManifest, load_manifest
from .schema import NeuralTextSample, SchemaError


_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_NAMES = ("manifest.jsonl", "dataset_manifest.jsonl", "features_manifest.jsonl")
_REQUIRED_METADATA = (
    "artifact_type",
    "authorization",
    "feature_storage",
    "feature_layout",
    "alignment_regime",
    "preprocessing_version",
)


class ZuCoFeatureSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ZuCoFeatureIssue:
    code: str
    severity: ZuCoFeatureSeverity
    message: str
    path: Path | None = None
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class ZuCoFeatureArtifactReport:
    root: Path
    manifest_path: Path | None = None
    manifest: DatasetManifest | None = None
    issues: tuple[ZuCoFeatureIssue, ...] = ()

    @property
    def errors(self) -> tuple[ZuCoFeatureIssue, ...]:
        return tuple(item for item in self.issues if item.severity == ZuCoFeatureSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ZuCoFeatureIssue, ...]:
        return tuple(item for item in self.issues if item.severity == ZuCoFeatureSeverity.WARNING)

    @property
    def passed(self) -> bool:
        return self.manifest is not None and not self.errors

    def require_valid(self) -> DatasetManifest:
        if not self.passed:
            codes = ", ".join(issue.code for issue in self.errors) or "missing manifest"
            raise ValueError(f"ZuCo precomputed feature artifact is invalid: {codes}")
        assert self.manifest is not None
        return self.manifest


class ZuCoPrecomputedFeatureAdapter:
    """Validate portable ZuCo feature artifacts without deserializing tensors.

    The artifact root must contain exactly one recognized JSONL manifest.  The
    manifest declares ``artifact_type=zuco_precomputed_features``, portable
    relative ``.pt`` feature references, SHA-256 checksums, and alignment
    metadata.  Tensor payload structure is intentionally deferred to a trusted
    training-time loader after this contract has been accepted.
    """

    name = "zuco_precomputed_features"
    dataset_id = "zuco_precomputed_features"

    def _root(self, source: str) -> Path:
        if not source.strip():
            raise ValueError("source must be a non-empty path")
        return Path(source).expanduser()

    @staticmethod
    def _feature_path(root: Path, uri: str) -> Path | None:
        candidate = Path(uri)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        return root / candidate

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def discover(self, source: str) -> ZuCoFeatureArtifactReport:
        """Inspect manifest fields and feature-file checksums without deserialization."""
        root = self._root(source)
        if not root.is_dir():
            return ZuCoFeatureArtifactReport(
                root=root,
                issues=(
                    ZuCoFeatureIssue(
                        "MISSING_ARTIFACT_DIRECTORY",
                        ZuCoFeatureSeverity.ERROR,
                        "artifact root does not exist or is not a directory",
                        root,
                    ),
                ),
            )
        candidates = tuple(path for name in _MANIFEST_NAMES if (path := root / name).is_file())
        if not candidates:
            return ZuCoFeatureArtifactReport(
                root=root,
                issues=(
                    ZuCoFeatureIssue(
                        "MISSING_FEATURE_MANIFEST",
                        ZuCoFeatureSeverity.ERROR,
                        "no recognized feature manifest JSONL file exists at artifact root",
                        root,
                    ),
                ),
            )
        if len(candidates) > 1:
            return ZuCoFeatureArtifactReport(
                root=root,
                issues=(
                    ZuCoFeatureIssue(
                        "AMBIGUOUS_FEATURE_MANIFEST",
                        ZuCoFeatureSeverity.ERROR,
                        "multiple recognized feature manifest names exist at artifact root",
                        root,
                    ),
                ),
            )
        manifest_path = candidates[0]
        try:
            manifest = load_manifest(manifest_path)
        except (OSError, SchemaError, ValueError) as error:
            return ZuCoFeatureArtifactReport(
                root=root,
                manifest_path=manifest_path,
                issues=(
                    ZuCoFeatureIssue(
                        "INVALID_FEATURE_MANIFEST",
                        ZuCoFeatureSeverity.ERROR,
                        f"manifest does not satisfy the canonical JSONL schema: {error}",
                        manifest_path,
                    ),
                ),
            )

        issues: list[ZuCoFeatureIssue] = []
        if not manifest.dataset_id.casefold().startswith("zuco"):
            issues.append(
                ZuCoFeatureIssue(
                    "INVALID_SOURCE_DATASET",
                    ZuCoFeatureSeverity.ERROR,
                    "feature manifest dataset_id must identify a ZuCo-derived artifact",
                    manifest_path,
                )
            )
        for field in _REQUIRED_METADATA:
            value = manifest.metadata.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(
                    ZuCoFeatureIssue(
                        "MISSING_FEATURE_METADATA",
                        ZuCoFeatureSeverity.ERROR,
                        f"required manifest metadata field is missing or invalid: {field}",
                        manifest_path,
                    )
                )
        if manifest.metadata.get("artifact_type") not in (None, "zuco_precomputed_features"):
            issues.append(
                ZuCoFeatureIssue(
                    "INVALID_ARTIFACT_TYPE",
                    ZuCoFeatureSeverity.ERROR,
                    "artifact_type must be 'zuco_precomputed_features'",
                    manifest_path,
                )
            )
        if manifest.metadata.get("feature_storage") not in (None, "torch_pt"):
            issues.append(
                ZuCoFeatureIssue(
                    "INVALID_FEATURE_STORAGE",
                    ZuCoFeatureSeverity.ERROR,
                    "feature_storage must be 'torch_pt' for this adapter",
                    manifest_path,
                )
            )
        if manifest.information_access.alignment_source == "unknown":
            issues.append(
                ZuCoFeatureIssue(
                    "MISSING_ALIGNMENT_METADATA",
                    ZuCoFeatureSeverity.ERROR,
                    "information_access.alignment_source must be declared",
                    manifest_path,
                )
            )
        if not manifest.samples:
            issues.append(
                ZuCoFeatureIssue(
                    "MISSING_FEATURE_SAMPLES",
                    ZuCoFeatureSeverity.ERROR,
                    "feature manifest has no canonical sample rows",
                    manifest_path,
                )
            )
        for sample in manifest.samples:
            self._validate_sample(root, sample, issues)
        return ZuCoFeatureArtifactReport(root, manifest_path, manifest, tuple(issues))

    def _validate_sample(
        self,
        root: Path,
        sample: NeuralTextSample,
        issues: list[ZuCoFeatureIssue],
    ) -> None:
        path = self._feature_path(root, sample.signal.uri)
        if path is None:
            issues.append(
                ZuCoFeatureIssue(
                    "NONPORTABLE_FEATURE_REFERENCE",
                    ZuCoFeatureSeverity.ERROR,
                    "feature URI must be a relative path contained in artifact root",
                    sample_id=sample.sample_id,
                )
            )
            return
        if path.suffix.casefold() != ".pt":
            issues.append(
                ZuCoFeatureIssue(
                    "INVALID_FEATURE_EXTENSION",
                    ZuCoFeatureSeverity.ERROR,
                    "feature URI must reference a .pt tensor artifact",
                    path,
                    sample.sample_id,
                )
            )
        expected = sample.signal.checksum_sha256
        if expected is None:
            issues.append(
                ZuCoFeatureIssue(
                    "MISSING_FEATURE_CHECKSUM",
                    ZuCoFeatureSeverity.ERROR,
                    "feature reference needs a SHA-256 checksum",
                    path,
                    sample.sample_id,
                )
            )
        elif _CHECKSUM_PATTERN.fullmatch(expected) is None:
            issues.append(
                ZuCoFeatureIssue(
                    "INVALID_FEATURE_CHECKSUM",
                    ZuCoFeatureSeverity.ERROR,
                    "feature checksum must be a lowercase SHA-256 hex digest",
                    path,
                    sample.sample_id,
                )
            )
        if not path.is_file():
            issues.append(
                ZuCoFeatureIssue(
                    "MISSING_FEATURE_FILE",
                    ZuCoFeatureSeverity.ERROR,
                    "referenced .pt feature file does not exist",
                    path,
                    sample.sample_id,
                )
            )
            return
        if path.stat().st_size == 0:
            issues.append(
                ZuCoFeatureIssue(
                    "INVALID_FEATURE_CONTENT",
                    ZuCoFeatureSeverity.ERROR,
                    "referenced feature file is empty",
                    path,
                    sample.sample_id,
                )
            )
            return
        if expected is not None and _CHECKSUM_PATTERN.fullmatch(expected) is not None:
            if self._checksum(path) != expected:
                issues.append(
                    ZuCoFeatureIssue(
                        "FEATURE_CHECKSUM_MISMATCH",
                        ZuCoFeatureSeverity.ERROR,
                        "feature file bytes do not match the manifest checksum",
                        path,
                        sample.sample_id,
                    )
                )

    def validate(self, source: str) -> ZuCoFeatureArtifactReport:
        """Alias for :meth:`discover` that exposes validation intent."""
        return self.discover(source)

    def build_manifest(self, source: str) -> DatasetManifest:
        return self.discover(source).require_valid()

    def iter_samples(self, source: str) -> Iterator[NeuralTextSample]:
        yield from self.build_manifest(source).samples
