# Data format and information-access contract

Every sample records modality, paradigm, subject/session, neural signal,
channels and geometry, text, event timing, split group, and provenance. Every
artifact additionally records the information visible at training, validation,
and inference time.

The supported alignment labels are `none`, `trial`, `fixation`, `word`,
`keystroke`, and `phoneme`. A continuous result cannot derive tensor length,
mask, word count, or internal boundaries from gold text.

See `src/openthought2text/data/` for the versioned data contracts and audit API.
