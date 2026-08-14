"""Portable JSONL manifests for canonical neural-text samples."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schema import InformationAccess, NeuralTextSample, SCHEMA_VERSION, SchemaError


MANIFEST_KIND = "openthought2text.manifest"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Dataset-level provenance plus an immutable sequence of sample rows."""

    dataset_id: str
    samples: tuple[NeuralTextSample, ...]
    information_access: InformationAccess
    source_url: str | None = None
    license: str | None = None
    description: str | None = None
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise SchemaError("manifest.dataset_id must be a non-empty string")
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"unsupported manifest schema version: {self.schema_version!r}")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise SchemaError("sample_id values must be unique in a manifest")
        if any(sample.dataset_id != self.dataset_id for sample in self.samples):
            raise SchemaError("every sample.dataset_id must equal manifest.dataset_id")

    def header_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": MANIFEST_KIND,
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "information_access": self.information_access.to_dict(),
        }
        for name in ("source_url", "license", "description"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


def write_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    """Write a JSONL header followed by canonical sample rows atomically enough for local use."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest.header_dict(), sort_keys=True) + "\n")
        for sample in manifest.samples:
            stream.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")


def _lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise SchemaError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise SchemaError(f"manifest row at {path}:{line_number} is not an object")
            yield line_number, value


def load_manifest(path: str | Path) -> DatasetManifest:
    source = Path(path)
    rows = _lines(source)
    try:
        first_line, header = next(rows)
    except StopIteration as error:
        raise SchemaError(f"manifest is empty: {source}") from error
    if header.get("kind") != MANIFEST_KIND:
        raise SchemaError(f"expected {MANIFEST_KIND!r} header at {source}:{first_line}")
    try:
        information_access = InformationAccess.from_dict(header["information_access"])
        samples = tuple(NeuralTextSample.from_dict(row) for _, row in rows)
        return DatasetManifest(
            dataset_id=str(header["dataset_id"]),
            samples=samples,
            information_access=information_access,
            source_url=header.get("source_url"),
            license=header.get("license"),
            description=header.get("description"),
            schema_version=str(header.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(header.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SchemaError(f"invalid manifest {source}") from error


def iter_manifest_samples(paths: Iterable[str | Path]) -> Iterator[NeuralTextSample]:
    """Yield samples from manifests while preserving each manifest's local order."""
    for path in paths:
        yield from load_manifest(path).samples
