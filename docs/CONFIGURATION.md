# Configuration

Configuration is versioned JSON in four layers:

- `configs/data/`: source, sampling, alignment access, and artifact settings.
- `configs/model/`: encoder, tokenizer, decoder, and head dimensions.
- `configs/task/`: paradigm, subject/split protocol, text constraint, and controls.
- `configs/experiment/`: a concrete named combination with a seed.

Every real run persists its resolved version of these layers in `RunManifest`.
Control conditions must be declared before evaluation; a report cannot add them
retroactively after seeing test results.
# Configuration

Checked-in JSON files are reviewable planning defaults, not permission to train
on a participant dataset. A real run requires a validated dataset card, release
bundle, authorized preflight plan, derived split plan, and target-free execution
spec before the configuration can be resolved into a run manifest.

`configs/model/continuous_conformer_tiny.json` uses the public
`NeuralToTextModelConfig` field names. Its filename is retained as the planned
continuous-baseline profile; it does not claim a completed Conformer training
run. `configs/experiment/zuco_tiny_alignment.json` references the
inventory-only `zuco_discovery` adapter and is explicitly marked
`preflight_required_no_participant_data_in_repository`.

Configuration values that affect splits, information access, tokenizer fitting,
normalization, model architecture, decoder controls, or reporting must be
written into the resolved run manifest and bound by provenance before metrics
are interpreted.
