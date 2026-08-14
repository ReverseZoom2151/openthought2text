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

__all__ = [
    "AdapterRegistry",
    "AuditFinding",
    "AuditReport",
    "DatasetAdapter",
    "DatasetManifest",
    "InformationAccess",
    "Modality",
    "NeuralTextSample",
    "PretrainingExposure",
    "SignalReference",
    "TextTarget",
    "TimeInterval",
    "SyntheticNeuralTextAdapter",
    "SyntheticValidationReport",
    "audit_splits",
    "load_manifest",
    "write_manifest",
]
