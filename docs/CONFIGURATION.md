# Configuration

Configuration is versioned JSON in four layers:

- `configs/data/`: source, sampling, alignment access, and artifact settings.
- `configs/model/`: encoder, tokenizer, decoder, and head dimensions.
- `configs/task/`: paradigm, subject/split protocol, text constraint, and controls.
- `configs/experiment/`: a concrete named combination with a seed.

Every real run persists its resolved version of these layers in `RunManifest`.
Control conditions must be declared before evaluation; a report cannot add them
retroactively after seeing test results.
