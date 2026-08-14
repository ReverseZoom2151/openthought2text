# Release checklist

This checklist distinguishes an implementation release from a scientific
benchmark claim. Checking a box does not authorize broader claims than its
evidence supports.

## Software release

- [ ] Supported Python-version CI is green.
- [ ] Package installation and synthetic CLI smoke paths pass.
- [ ] Versioned configuration, dataset-card, model-card, and provenance
  schemas are checked in.
- [ ] No participant data, trained checkpoint, or unlicensed reference asset is
  included in the distributable package.

## Real-data benchmark release

- [ ] Authorized dataset artifact and validated dataset card are present.
- [ ] Derived split plan, information-access contract, and preprocessing
  artifact are checksummed and saved.
- [ ] Generation is target-free and passes signature plus label-invariance
  audit.
- [ ] Full, shuffled, zero/noise, mask, length, timing, and surrogate controls
  have paired saved predictions.
- [ ] Subject-aware uncertainty, permutation results, error taxonomy, and
  occlusion results are saved.
- [ ] Release gate passes; model card binds the exact evaluation and provenance
  artifacts.

If any real-data item is missing, publish an implementation status only—not a
neural-decoding performance claim.
