<h1 align="center">OpenThought2Text</h1>

<p align="center"><strong>Evidence-audited neural activity to text research framework</strong></p>

An in-progress, test-driven research implementation for decoding text associated
with constrained recorded language tasks. OpenThought2Text is building a
reproducible EEG/MEG/intracortical research pipeline around leakage-safe data
contracts, neural representations, target-free decoding, and explicit evidence
controls.

It is not a demonstration of unrestricted mind reading: this repository
contains no trained checkpoints, benchmark claims, clinical capability, or
evidence that arbitrary private thoughts can be decoded.

## What is implemented

- A clean-room, Apache-2.0 Python package with a command-line foundation,
  reproducibility manifest, composable losses, synthetic workflow, and CI.
- Canonical neural-text samples, dataset manifests, adapter registry, and split
  auditing for duplicate text/groups, continuous-window overlap, side-channel
  metadata, and declared pretraining overlap.
- Neural model foundations: continuous neural encoder, timing metadata,
  coordinate-aware and graph montage adapters, subject adapters, VQ tokenizer
  diagnostics, semantic query pooling, contrastive alignment, anchors, CTC,
  and a target-free autoregressive decoder.
- Evaluation foundations: CER/WER, retrieval metrics, grounded-gain reporting,
  target-free signature/label-invariance audits, and shuffled/zero/noise/mask/
  length/timing/surrogate controls, faithfulness suites, paired statistics,
  channel/time occlusion, saved prediction artifacts, and reports.
- A synthetic end-to-end workflow: artifact preparation, train-only
  normalization, masked variable-length batches, supervised training steps,
  checkpoint provenance, and CLI-based reporting.
- Project governance, model/dataset-card templates, responsible-use guidance,
  and a research archive that stays outside the distributable package.

## What still requires execution

Real participant datasets, real ZuCo preprocessing artifacts, trained weights,
full ZuCo results, cross-subject evaluation, continuous-decoding performance,
foundation-model reproduction, and any claims of neural decoding quality remain
separate from the offline development path. Read the
[implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md),
[capability status](docs/CAPABILITIES.md), and
[paper-fidelity status](docs/PAPER_FIDELITY.md) before spending compute.

## Install

```bash
git clone https://github.com/ReverseZoom2151/openthought2text.git
cd openthought2text

# Offline development and contract checks
python -m pip install -e '.[dev]'
python -m pytest -q

# Install a matching CPU/CUDA PyTorch build before real training.
```

Python 3.10–3.12 is supported. The default test path uses synthetic data only;
participant datasets and runtime-heavy research dependencies are deliberately
not bundled.

## Core workflow

```text
recorded neural signal
  → validated dataset artifact and information-access manifest
  → leakage-safe subject/stimulus/recording split
  → modality-aware neural encoder and semantic representation
  → target-free decoder or candidate-evidence scorer
  → controls, saved predictions, and reproducible report
```

Useful commands:

```bash
# Dataset and split entry points
ott data discover --dataset zuco_v1 --root /path/to/zuco
ott data validate --dataset zuco_v1 --root /path/to/zuco
ott data card-validate --card /path/to/dataset_card.json
ott splits audit --artifact /path/to/artifact --protocol loso_subject_unique_text
ott splits build --manifest /path/to/source.jsonl --output /path/to/derived.jsonl \
  --protocol loso_subject_unique_text --held-out-subject participant-17 --seed 7

# Generation and grounding audits
ott evaluate audit-generation --checkpoint /path/to/checkpoint
ott evaluate compare-controls --run /path/to/run \
  --controls full,shuffled,zero,noise,mask,length,timing,lm_only

# Synthetic package smoke path
python examples/synthetic_workflow.py
python examples/synthetic_experiment.py

# Reproducible synthetic run artifacts (checkpoint, tokenizer, normalizer,
# and target-free test predictions). The output directory must not exist.
ott data prepare --dataset synthetic --root /tmp/ott-synthetic
ott train synthetic --root /tmp/ott-synthetic --output /tmp/ott-synthetic-run
```

## Documentation

- [Implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md)
- [Full master implementation plan](docs/MASTER_PLAN.md)
- [Capability status](docs/CAPABILITIES.md)
- [Paper-fidelity status](docs/PAPER_FIDELITY.md)
- [Data format and information-access contract](docs/DATA_FORMAT.md)
- [Synthetic dataset card](dataset_cards/synthetic.json)
- [Configuration](docs/CONFIGURATION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md) and [dependency policy](docs/DEPENDENCY_POLICY.md)
- [Architecture](docs/architecture/README.md), [scope](docs/ethics/scope.md), and
  [workspace layout](docs/WORKSPACE_LAYOUT.md)

## Contributing

Run `python -m pytest -q` before submitting changes. Keep claims tied to saved,
versioned artifacts; keep generation target-free; and update capability,
paper-fidelity, model-card, and dataset-card records whenever a research path
changes status.

## License and citation

Released under the [Apache-2.0 License](LICENSE). This repository is an
independent implementation informed by the research cited in the documentation;
reference code and papers are not redistributed as part of this package.

See [CITATION.cff](CITATION.cff) for software citation metadata.
