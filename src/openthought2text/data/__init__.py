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
from .zuco_precomputed import (
    ZuCoFeatureArtifactReport,
    ZuCoFeatureIssue,
    ZuCoFeatureSeverity,
    ZuCoPrecomputedFeatureAdapter,
)
from .splits import (
    SplitPlan,
    SplitProtocol,
    SplitValidationReport,
    SplitViolation,
    build_split_plan,
    validate_split_plan,
)
from .tokenizer import (
    TrainTextTokenizer,
    UnknownTokenPolicy,
    fit_train_text_tokenizer,
    load_train_text_tokenizer,
    tokenize_text,
    write_train_text_tokenizer,
)
from .json_signals import (
    ManifestSplit,
    load_json_tensor_samples,
    select_split_samples,
)
from .dataset_card import (
    DatasetCard,
    DatasetCardIssue,
    DatasetCardValidationReport,
    load_dataset_card,
    validate_dataset_card,
    write_dataset_card,
)
from .task_adapters import (
    Brain2QwertyDiscoveryAdapter,
    LabelAccess,
    T15DiscoveryAdapter,
    TaskAdapterRequirements,
    TaskDiscoveryIssue,
    TaskDiscoveryReport,
    TypedTaskDiscoveryAdapter,
)
from .authorized_features import (
    AUTHORIZED_FEATURE_KIND,
    AUTHORIZED_FEATURE_VERSION,
    ArtifactAuditIssue,
    ArtifactAuditReport,
    AuthorizedFeatureArtifact,
    AuthorizedFeatureMapping,
    audit_authorized_json_features,
    load_authorized_json_features,
)
from .release_bundle import (
    RELEASE_BUNDLE_KIND,
    RELEASE_BUNDLE_VERSION,
    DatasetReleaseBundle,
    ReleaseArtifactReference,
    ReleaseBundleAuditReport,
    ReleaseBundleIssue,
    audit_dataset_release_bundle,
    build_dataset_release_bundle,
    write_dataset_release_bundle,
)
from .montage import MontageProvenance, NamedMontage, select_named_montage
from .augmentation import (
    NeuralAugmentationConfig,
    augment_neural_tensor_batch,
    augment_tensor_backed_samples,
)
from .continuous_chunks import (
    SIGNAL_TIMELINE_ALIGNMENT,
    ContinuousChunkProvenance,
    ContinuousChunkView,
    build_continuous_chunk_view,
)
from .sensor_layout import (
    SensorDefinition,
    SensorLayout,
    SensorLayoutTensor,
    SensorType,
    sensor_layout_to_tensor,
)

__all__ = [
    "AdapterRegistry",
    "AUTHORIZED_FEATURE_KIND",
    "AUTHORIZED_FEATURE_VERSION",
    "AuditFinding",
    "AuditReport",
    "ArtifactAuditIssue",
    "ArtifactAuditReport",
    "AuthorizedFeatureArtifact",
    "AuthorizedFeatureMapping",
    "Brain2QwertyDiscoveryAdapter",
    "ChannelNormalizer",
    "ContinuousChunkProvenance",
    "ContinuousChunkView",
    "DatasetAdapter",
    "DatasetCard",
    "DatasetCardIssue",
    "DatasetCardValidationReport",
    "DatasetManifest",
    "DatasetReleaseBundle",
    "InformationAccess",
    "LabelAccess",
    "ManifestSplit",
    "MontageProvenance",
    "Modality",
    "NeuralTextSample",
    "NeuralAugmentationConfig",
    "NamedMontage",
    "NeuralTensorBatch",
    "PretrainingExposure",
    "PreparedArtifactManifest",
    "PreparedTensorRecord",
    "RELEASE_BUNDLE_KIND",
    "RELEASE_BUNDLE_VERSION",
    "ReleaseArtifactReference",
    "ReleaseBundleAuditReport",
    "ReleaseBundleIssue",
    "SignalReference",
    "SIGNAL_TIMELINE_ALIGNMENT",
    "SensorDefinition",
    "SensorLayout",
    "SensorLayoutTensor",
    "SensorType",
    "SplitPlan",
    "SplitProtocol",
    "SplitValidationReport",
    "SplitViolation",
    "TextTarget",
    "TimeInterval",
    "SyntheticNeuralTextAdapter",
    "SyntheticValidationReport",
    "TensorBackedSample",
    "T15DiscoveryAdapter",
    "TaskAdapterRequirements",
    "TaskDiscoveryIssue",
    "TaskDiscoveryReport",
    "TrainTextTokenizer",
    "TypedTaskDiscoveryAdapter",
    "UnknownTokenPolicy",
    "VariableLengthTensorDataset",
    "ZuCoDiscoveryAdapter",
    "ZuCoDiscoveryReport",
    "ZuCoFeatureArtifactReport",
    "ZuCoFeatureIssue",
    "ZuCoFeatureSeverity",
    "ZuCoLayoutIssue",
    "ZuCoLayoutSeverity",
    "ZuCoPrecomputedFeatureAdapter",
    "ZuCoTaskInventory",
    "audit_splits",
    "augment_neural_tensor_batch",
    "augment_tensor_backed_samples",
    "audit_authorized_json_features",
    "audit_dataset_release_bundle",
    "build_prepared_artifact_manifest",
    "build_continuous_chunk_view",
    "build_dataset_release_bundle",
    "build_split_plan",
    "collate_tensor_backed_samples",
    "fit_train_channel_normalizer",
    "fit_train_text_tokenizer",
    "load_manifest",
    "load_dataset_card",
    "load_json_tensor_samples",
    "load_authorized_json_features",
    "load_prepared_artifact_manifest",
    "load_train_text_tokenizer",
    "tensor_checksum",
    "tokenize_text",
    "select_split_samples",
    "select_named_montage",
    "sensor_layout_to_tensor",
    "write_manifest",
    "write_dataset_card",
    "write_dataset_release_bundle",
    "write_prepared_artifact_manifest",
    "write_train_text_tokenizer",
    "validate_split_plan",
    "validate_dataset_card",
]
