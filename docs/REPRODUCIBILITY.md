# Reproducibility

Each trainable run must save its resolved configuration, seed, code revision,
artifact and split checksums, hardware/precision, parameter counts, prediction
files, and selection rule. Normalization and codebooks use training partitions
only unless a declared pretrained-transfer manifest proves no evaluation overlap.

The CI/test path has no participant-data dependency. Real dataset preparation
and model training are explicit opt-in operations.
