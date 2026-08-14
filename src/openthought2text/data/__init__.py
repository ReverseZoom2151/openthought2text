"""Dataset contracts, manifests, adapters, and split-integrity audits.

This package deliberately has no dependency on a training framework.  Every
dataset is normalized to the same metadata contract before it can reach a
model or a split definition.
"""

from .adapters import AdapterRegistry, DatasetAdapter
from .audit import AuditFinding, AuditReport, PretrainingExposure, audit_splits
from .manifest import DatasetManifest, load_manifest, write_manifest
from .schema import (
    InformationAccess,
    Modality,
    NeuralTextSample,
    SignalReference,
    TextTarget,
    TimeInterval,
)
from .synthetic import SyntheticNeuralTextAdapter, SyntheticValidationReport
from .prepared import (
    ChannelNormalizer,
    PreparedArtifactManifest,
    PreparedTensorRecord,
    TensorBackedSample,
    build_prepared_artifact_manifest,
    fit_train_channel_normalizer,
    load_prepared_artifact_manifest,
    tensor_checksum,
    write_prepared_artifact_manifest,
)
from .batching import (
    NeuralTensorBatch,
    VariableLengthTensorDataset,
    collate_tensor_backed_samples,
)
from .zuco import (
    ZuCoDiscoveryAdapter,
    ZuCoDiscoveryReport,
    ZuCoLayoutIssue,
    ZuCoLayoutSeverity,
    ZuCoTaskInventory,
)

__all__ = [
    "AdapterRegistry",
    "AuditFinding",
    "AuditReport",
    "ChannelNormalizer",
    "DatasetAdapter",
    "DatasetManifest",
    "InformationAccess",
    "Modality",
    "NeuralTextSample",
    "NeuralTensorBatch",
    "PretrainingExposure",
    "PreparedArtifactManifest",
    "PreparedTensorRecord",
    "SignalReference",
    "TextTarget",
    "TimeInterval",
    "SyntheticNeuralTextAdapter",
    "SyntheticValidationReport",
    "TensorBackedSample",
    "VariableLengthTensorDataset",
    "ZuCoDiscoveryAdapter",
    "ZuCoDiscoveryReport",
    "ZuCoLayoutIssue",
    "ZuCoLayoutSeverity",
    "ZuCoTaskInventory",
    "audit_splits",
    "build_prepared_artifact_manifest",
    "collate_tensor_backed_samples",
    "fit_train_channel_normalizer",
    "load_manifest",
    "load_prepared_artifact_manifest",
    "tensor_checksum",
    "write_manifest",
    "write_prepared_artifact_manifest",
]
