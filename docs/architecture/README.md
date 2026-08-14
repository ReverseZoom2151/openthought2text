# Architecture and information flow

OpenThought2Text is a research framework for evaluating whether a model uses
permitted neural signal evidence to produce text or constrained candidates. It
does not represent a validated mind-reading system, and the repository ships
no participant data or pretrained decoding checkpoint.

## Core path

```text
Dataset card + authorized source
        |
        v
Manifest / information-access contract / leakage audit
        |
        v
Named montage -> train-only normalization -> masked batch
        |
        v
Neural encoder -> optional bottleneck or RVQ -> neural evidence
        |                                      |
        |                                      +--> CTC / anchor / retrieval heads
        v
Target-free decoder or authorized candidate ranker
        |
        v
Prediction records -> full/control evaluation -> release gate
```

The canonical sample and manifest contracts live in `openthought2text.data`.
They retain subject/session/stimulus grouping, modality, signal references,
sensor geometry, declared alignment access, and provenance. Split construction
and audit must happen before fitting normalization, vocabulary, any retrieval
index, or a learned model.

## Inference boundary

Production-style inference functions accept neural tensors, masks, sensor
coordinates, and explicitly authorized candidate inputs only. They must not
accept target token IDs, reference text, target lengths, stimulus lookup keys,
or gold timing. Teacher-forced training methods are distinct from generation
methods. The prediction writer and faithfulness suite inspect that boundary and
run full, shuffled, zero, noise, mask, length, timing, and phase-surrogate
controls.

`NeuralToTextModel.generate` is the neural-to-token generation boundary.
`EvidenceGroundedCandidateRanker.forward` ranks an explicitly supplied,
authorized candidate set without labels. TorchScript export is intentionally
narrower: it exports only the fixed-input, tensor-only
`neural_encoder_evidence_v1` encoder-evidence path and records the exact input
signature in its sidecar. Autoregressive generation is not claimed portable by
that export.

## Model families

The shared encoder output is `NeuralEncoderOutput(features, mask, timing)`.
All encoder implementations must zero masked values, preserve a token-validity
mask, and report timing derived from signal sampling rather than target text.
The package includes coordinate and graph channel fusion, subject adapters,
continuous convolutional encoding, semantic query pooling, residual
quantization, CTC and semantic-anchor heads, a target-free decoder, and an
evidence-grounded candidate ranker. Baseline and research-family modules are
kept as separately configurable implementations so comparisons do not silently
share data access or language-model assistance.

## Evidence and release flow

No score alone supports a claim. Evaluation artifacts bind saved predictions to
their references, control condition, split, provenance, and metrics. The
release gate requires paired sample identities, an explicit target-free audit,
declared controls, and grounded-gain/neural-contribution evidence before a
model card can mark a result as supported. Dataset release bundles bind cards,
manifests, split plans, and authorized feature descriptors by SHA-256.

See [the master plan](../MASTER_PLAN.md) for staged implementation and
[the inference contract](../reproducibility/inference-contract.md) for the
non-negotiable information-access rules.
