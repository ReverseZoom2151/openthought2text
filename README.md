# OpenThought2Text

OpenThought2Text is an open research framework for **constrained neural-activity-to-text decoding**. It supports recorded EEG, MEG, and intracortical language-task signals while treating evaluation validity, neural grounding, and participant privacy as first-class requirements.

It does not claim to decode arbitrary private thoughts or internal monologue.

## Project principles

- Every result names its modality, paradigm, alignment regime, subject protocol, text constraint, and decoding mode.
- Teacher-forced logits are training diagnostics, never headline generation results.
- Inference paths are target-free; tests verify that changing labels cannot change a prediction.
- Full-signal results are compared with shuffled, zero, noise, mask-only, length-only, timing-only, and language-prior controls.
- Reference research is preserved outside the package under [`../references/`](../references/); implementation is clean-room unless license terms explicitly allow reuse.

## Status

The project is in the implementation phase. The governing roadmap is the [workspace master plan](../THOUGHT_TO_TEXT_MASTER_PLAN.md). The first build target is a reproducible ZuCo reading-EEG benchmark with explicit word-aligned and continuous/fixation-free tracks.

## Quick start

```bash
cd openthought2text
python -m pip install -e '.[dev]'
ott --help
pytest
```

## Safety and scope

Outputs are experimental task-associated decodes, not ground truth mental content. Do not use this project for clinical, forensic, employment, educational, surveillance, or high-stakes decision-making purposes.
