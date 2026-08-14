# Capability status

## Implemented offline foundations

- Canonical data, manifest, adapter-registry, train-only normalization and
  vocabulary artifacts, portable JSON-signal loading, prepared-artifact,
  masked variable-length batching, named montage selection, heterogeneous
  sensor layouts, fixed-duration signal-only chunks, deterministic safe
  augmentations, patching contracts, and split-audit contracts.
- Synthetic end-to-end data preparation/validation and safe discovery contracts
  for raw-layout and precomputed-feature ZuCo artifacts.
- Continuous encoder, coordinate-aware and graph montage adapters, subject
  adaptation, GRU/ChannelNet/compact-Conformer baselines, VQ diagnostics,
  semantic queries, contrastive alignment, anchor, CTC, CTC beam search,
  target-free sequence decoding, masked reconstruction/consistency objectives,
  evidence-factorized candidate scoring, and scoped TorchScript export.
- Text/retrieval metric primitives, target-free inference audits, neural
  grounding controls, faithfulness suites, bootstrap/permutation statistics,
  channel/time occlusion, BLEU/ROUGE-L/unigram-METEOR-style metrics, CPU timing
  measurement, saved predictions, and evaluation reports.
- Config-validated model factory, auditable residual bottleneck, constrained
  candidate ranker, supervised epoch runner, checkpoint compatibility checks,
  target-free token-prediction serialization, hash-bound provenance, release
  gate, model/reproduction/multi-seed artifacts, dataset release bundles,
  authorized-data preflight plans, and target-free execution specifications.
- A regression-tested synthetic trace spanning generated signals, train-only
  fitting, training, target-free held-out generation, and checkpoint creation.
- An educational static synthetic interface and an inventory-only CLI for ZuCo,
  Brain2Qwerty, and T15 layouts. Neither parses participant recordings.

## Not yet validated experimentally

- Any performance on ZuCo, Brain2Qwerty, T15, MEG, or silent-articulation data.
- Any generated text quality, cross-subject generalization, latency, or
  foundation-model improvement.
- Any clinical, forensic, private-thought, or real-world assistive claim.

## Required evidence before a capability changes status

Saved predictions, leakage-safe split manifests, target-free generation audit,
full control suite, subject-level uncertainty, a dataset card, and a model card.
For a real run, the bindings additionally need an authorized preflight plan and
an execution specification with declared controls and permitted inference fields.
