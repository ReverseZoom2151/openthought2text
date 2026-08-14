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
