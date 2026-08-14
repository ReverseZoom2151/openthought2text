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

The detailed master plan remains maintained in the parent research workspace;
release-facing decisions are mirrored here as implementation work lands.
