# Architecture decisions

The implementation follows the master plan's two-tier design:

1. A reproducible continuous-Conformer ZuCo baseline with strict controls.
2. Optional research wrappers for LaBraM, BrainMagick, BioCodec/MEG-XL, and evidence-factorized decoding.

Every wrapper must preserve token timing, subject protocol, alignment access, and an information-access manifest.
