# Adding a dataset integration

This template is for a metadata and adapter contribution. It does not grant
access to a dataset, authorize redistribution, or establish a benchmark claim.
Do not commit participant recordings, raw neural arrays, participant metadata,
access tokens, trained checkpoints, or restricted source artifacts.

## 1. State the scope before writing code

Fill in the integration issue or pull request description:

- Dataset identifier: `[dataset_id]`
- Intended task and modality/modalities: `[task]`, `[modality]`
- Contribution type: `[discovery-only | authorized-precomputed-features | synthetic fixture]`
- Source license and consent/access basis: `[short statement and authoritative link]`
- Does this change parse raw participant data? It must be `no` unless that
  parsing is explicitly authorized, reviewed, and implemented outside a public
  fixture path.

An adapter must not imply that a dataset is supported simply because its name
appears in documentation. Clearly distinguish a layout/discovery contract from
an authorized artifact loader and from a runnable benchmark.

## 2. Provide a strict dataset card

Create a JSON dataset card using the existing card contract. It must disclose
the source, license, consent, access condition, modality, split protocol, and
preprocessing description. Validate its checksum and keep the card free of
participant identifiers beyond the public, approved dataset-level metadata.

Before proceeding, the card must be ready under `load_dataset_card` /
`validate_dataset_card`; a missing disclosure is a release blocker.

## 3. Bind an authorized release, not raw data

For an authorized precomputed-feature integration, construct a release bundle
that binds all of the following by checksum:

1. The validated dataset card.
2. The canonical manifest.
3. The derived split plan.
4. The authorized feature descriptor.
5. The manifest information-access contract.

Use local relative metadata references only. The release bundle and feature
descriptor may be checked in only when they contain no restricted participant
payloads. Raw files remain in the authorized environment; never substitute a
path to an individual participant recording for an artifact binding.

## 4. Declare information access and splits

Document exactly what training, validation, and inference receive. Inference
must not receive target text or text context unless the project explicitly
labels that as leakage. State the alignment source and preserve it in the
manifest/release binding.

Choose and materialize a named split protocol. Run `validate_split_plan` and
the split audit before publishing any results. Audit duplicate text, group or
subject overlap, continuous interval overlap, and known pretraining exposure.
Do not choose chunks, sequence length, masks, word counts, or boundaries from
gold target text in a continuous setting.

## 5. Create an authorized preflight plan

Create a JSON preflight plan that binds the card, release bundle, and split
plan; declares a non-path source-root identifier, authorization identifier,
inference-access contract, and requested protocols. The plan is metadata-only:
it must not load signals. Its audit must pass before the authorized environment
is used.

## 6. Add tests and review hooks

Every integration needs small synthetic filesystem fixtures only. Add or extend
tests that cover:

- card schema/checksum/disclosure validation;
- adapter discovery and explicit missing or ambiguous layout reports;
- release-bundle and preflight round trip plus checksum tampering;
- information-access and split-plan validation, including a leakage case;
- the safe loader's path, JSON payload, checksum, and train-only audit checks;
- an assertion that no participant data fixture or raw-data parser is needed.

Suggested hooks are `tests/unit/test_data_<dataset>.py` for discovery/contract
tests and the existing `test_data_audit.py`, `test_data_release_bundle.py`, and
`test_data_preflight.py` patterns for shared gates. Run the focused test file
and the full `tests/unit/test_data*.py` suite.

## Pull-request handoff

Include the completed JSON checklist from this directory, links to the dataset
card and authorized metadata artifacts, tests run, and a statement that no
participant data was committed. If access, consent, authorization, or a split
audit is unresolved, stop at an implementation/discovery contribution and do
not report real-data decoding metrics.
