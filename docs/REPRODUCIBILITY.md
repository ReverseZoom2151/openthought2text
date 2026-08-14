# Reproducibility

Each trainable run must save its resolved configuration, seed, code revision,
artifact and split checksums, hardware/precision, parameter counts, prediction
files, and selection rule. Normalization, vocabulary fitting, and codebooks use
training partitions only unless a declared pretrained-transfer manifest proves
no evaluation overlap.

## Derived splits

Never edit an input manifest in place. Build an explicit derived manifest and
its adjacent, deterministic split-plan artifact instead:

```bash
ott splits build \
  --manifest /path/to/source_manifest.jsonl \
  --output /path/to/derived_loso.jsonl \
  --protocol loso_subject_unique_text \
  --held-out-subject participant-17 \
  --seed 7
```

The command refuses to overwrite either output and validates continuous-window,
text, session, and held-out-subject constraints required by the selected
protocol. The derived manifest records its source and the full plan; the
sidecar is named by replacing `.jsonl` with `.split_plan.json`.

## Binding an evaluation to its inputs

Before interpreting a result, create a `RunArtifactProvenance` report. It
contains SHA-256 bindings for the model, checkpoint, data manifest, split plan,
and resolved configuration, plus code revision and an explicit declaration of
what was visible at inference. Its `binding_sha256` is recomputed when loaded;
missing, placeholder, or altered bindings are rejected. This report is a
necessary traceability record, not evidence that a model is accurate.

The CI/test path has no participant-data dependency. Real dataset preparation
and model training are explicit opt-in operations.
