# Implementation roadmap

The project uses a 20-week credible core release followed by an eight-week
frontier expansion.

The core path is ZuCo-first: dataset validation, preprocessing, baseline
retrieval, continuous neural encoding, semantic anchors, target-free generation,
faithfulness controls, and strict held-out-subject/unique-text reporting.

The research expansion adds corrected literature baselines, LaBraM,
BrainMagick, fixation-free EEG, BioCodec/MEG-XL, ActiveLBLM, and
evidence-factorized decoding. Each addition must beat a compute-matched
continuous baseline on strict neural contribution rather than fluency alone.

See the [full master implementation plan](MASTER_PLAN.md) for every research
traceability decision, architecture choice, stage, release gate, and weekly
deliverable.

## Current implementation boundary

The offline foundation is implemented and exercised on a deterministic,
non-participant synthetic artifact: manifest validation, portable signal
loading, leakage-aware split construction, train-only normalization and text
vocabulary fitting, fixed-duration continuous views, baseline/encoder families,
self-supervision and discrete-token interfaces, target-free generation and CTC
beam search, prediction serialization, checkpoint/provenance records,
factorized evidence scoring, and evidence gating.

The next research milestone is deliberately not “train a bigger model.” It is
to prepare an authorized real-data artifact with a dataset card, release bundle,
authorized preflight plan, a derived held-out-subject/unique-text split plan,
an explicit information-access contract, target-free execution specification,
and saved full/control predictions. Only then can a model-card claim become
supported by the release gate. Until those artifacts exist, no real dataset
result or neural-decoding quality claim is implemented.
