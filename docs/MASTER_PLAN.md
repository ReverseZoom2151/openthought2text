# OpenThought2Text: Master Implementation Plan

Status: pre-implementation planning document, revised after full workspace expansion
Direction: neural activity to text (`brain signal -> text`)
Initial release target: reproducible non-invasive EEG-to-text benchmark and model
Planning horizon: 20-week credible v0.1 release plus an 8-week frontier expansion

Revision coverage: the original workspace plus CET-MAE/E2T-PTR, DeWave, NeuSpeech's corrected EEG-to-text evaluation, LaBraM, BrainMagick, MEG-XL/BioCodec, ActiveLBLM, and HuthLab semantic decoding. Paper revisions are tracked separately when multiple versions exist.

## 1. Direction correction and project definition

This is a **thought-to-text / neural-activity-to-text** project. The input is a recorded neural signal and the output is text. It is not the text-to-neural generator described in the earlier draft.

The scientifically accurate project claim is:

> OpenThought2Text is an open framework for decoding text associated with constrained, recorded language tasks from EEG, MEG, or intracortical neural activity.

The project must not claim to decode arbitrary, private, unprompted thoughts. None of the datasets in this workspace supports that claim. The available paradigms measure neural activity during one of the following:

- reading known text;
- viewing known images;
- typing memorized sentences;
- attempted speech;
- motor or inner-speech command classification.

“Thought-to-text” can be the accessible project name, but the README, model cards, paper, API documentation, and demo must name the exact paradigm used by each checkpoint.

## 2. Product thesis

The workspace contains several strong but disconnected research directions. The open-source opportunity is not another one-off notebook. It is a common system with:

1. a standard neural-text sample schema;
2. reproducible dataset adapters and leakage-safe splits;
3. modality-specific neural front ends;
4. a shared semantic bottleneck;
5. task-appropriate text heads;
6. strict separation of neural evidence from language-model prior;
7. cross-subject evaluation and calibration;
8. faithfulness controls that expose hallucination and posterior collapse;
9. small, medium, and research-scale configurations;
10. model cards, dataset cards, tests, and a public benchmark.

The flagship model will be modular rather than pretending that reading EEG, typing MEG, and intracortical attempted speech are interchangeable.

### 2.1 Decisions introduced by the expanded workspace

The newly added work changes the plan in seven mandatory ways:

1. **Teacher-forced evaluation is not generation.** Teacher-forced loss and token accuracy are training diagnostics only. Headline text results must come from target-free autoregressive generation or a clearly defined CTC/RNN-T decoder.
2. **Alignment access is part of the task definition.** Word/fixation boundaries, word counts, keystroke times, and phoneme boundaries are useful oracles. Results using them cannot be compared directly with fixation-free continuous decoding.
3. **Noise must be compared against more than EEG.** Random input can retain sequence length, masks, event count, and timing. The benchmark therefore includes mask-only, length-only, timing-only, and fully uninformative controls.
4. **Foundation encoders become an explicit research lane.** LaBraM, BrainMagick, and MEG-XL motivate reusable neural tokenizers and cross-dataset encoders, but they do not replace a reproducible small-data baseline.
5. **Discrete neural tokens are a first-class experiment, not the default assumption.** LaBraM's VQ spectrum codes and MEG-XL's BioCodec RVQ codes must be compared with continuous Conformer features under identical splits.
6. **Brain evidence and language prior need independently inspectable scores.** HuthLab's semantic decoder motivates candidate generation by an LM followed by a separately measurable neural-evidence likelihood.
7. **Closed-vocabulary silent articulation is a distinct task.** ActiveLBLM is relevant to representation learning and silent-speech classification, but its 24-word setting must not be presented as open-vocabulary sentence decoding.

## 3. Release strategy

### Release 0.1: EEG reading-to-text

Primary dataset: ZuCo 1.0 and, when its access and preprocessing are validated, ZuCo 2.0.

Goals:

- load and validate word- and sentence-aligned EEG;
- reproduce simple baselines;
- train a cross-subject neural-text representation;
- decode semantic anchors and sentences;
- publish leakage-safe in-subject and leave-one-subject-out results;
- provide zero, shuffled, noise, mask-only, length-only, timing-only, surrogate-signal, and text-only controls.

Release 0.1 has two explicitly separated benchmark tracks:

- `reading_word_aligned`: fixation/word events are available to the model; this is an oracle-aligned semantic decoding task;
- `reading_continuous`: the model receives raw or regularly chunked EEG without gold word boundaries, word count, or fixation order at inference.

The word-aligned track ships first. The continuous track is allowed to remain experimental in v0.1, but its schema, controls, and baseline must already exist.

This is the most feasible open release because the workspace already contains ZuCo loading, preprocessing, GLIM, and GraphAlign implementations.

### Release 0.2: typed sentence production

Primary dataset: public Brain2Qwerty v1 SpanishBCBL data.

Goals:

- add EEG/MEG signal adapters;
- implement synchronous keystroke-aligned decoding;
- add character CTC and sentence-context refinement;
- evaluate EEG versus MEG without conflating the modalities.

The English Brain2Qwerty v2 dataset is currently described as embargoed in the local repository. The code architecture can inform the design, but the release cannot depend on unavailable data.

### Release 0.3: attempted speech

Primary dataset: T15 attempted-speech neural data used by the `brain-to-text-working` project.

Goals:

- support intracortical feature streams;
- predict phonemes with CTC or RNN-T;
- decode with an explicitly separate language model;
- reproduce phoneme error rate and word error rate baselines.

### Post-v0.1 research track, targeted as release 0.4: neural foundation and long-context models

This is an opt-in expansion after the Week 20 reproducible release, not a dependency of v0.1.

Goals:

- add LaBraM-compatible VQ spectrum tokenization and masked-code prediction;
- add a BrainMagick-style continuous convolutional encoder with coordinate-aware channel merger, subject layers, and brain-audio contrastive alignment;
- add BioCodec residual vector quantization and MEG-XL-style criss-cross temporal/spatial attention;
- evaluate frozen, linear-probe, partial-fine-tune, and from-scratch regimes;
- add continuous/fixation-free DeWave-style EEG-to-text evaluation;
- add optional MEG language datasets and long-context word retrieval;
- add ActiveLBLM silent-articulation classification as a clearly separate constrained task;
- publish compute-matched comparisons so scale is not confused with architectural merit.

### Release 1.0: unified benchmark and pretrained model family

Goals:

- shared data contract across all supported paradigms;
- per-modality encoders with a common semantic API;
- population, subject-adapted, and reduced-channel checkpoints;
- benchmark tables with uncertainty and permutation tests;
- stable Python API and command-line interface;
- documented extension interface for new datasets and modalities.

## 4. What “using all the workspace work” means

The project will use every relevant repository and paper as an architectural lesson, baseline, data source, validation method, or explicit negative result. It will not force unrelated code into the model, copy code whose license is incompatible, or treat low-quality prototypes as validated science.

### 4.1 Repository traceability matrix

| Workspace item | Decision | Concrete use in OpenThought2Text |
|---|---|---|
| `Thought2Text-main` | Adapt and independently reimplement | ChannelNet-style EEG encoder baseline; staged EEG/image/text training lesson; only-EEG, only-object, and chance controls; LLM prefix projection; NLG evaluation. Avoid treating image-caption knowledge as neural evidence. |
| `Thought2Text-main (1)` | Reference only | MEG preprocessing, ICA, MEG2VEC/TEXT2VEC decomposition, and exploratory notebook lessons. Do not use its broad medical or forensic claims. |
| `Thought2Text-16Ch-BCI-main` | Adapt | Named montage selection, learnable 16-to-high-density channel adapter, FP16 packaging, edge benchmark, model bundle, checksums, and synthetic smoke tests. |
| `GLIM-main` | Core research influence | Semantic summarization framing; prompt-free and noise-input evaluation; posterior-collapse prevention; EEG-text retrieval; strict unique-text splits; ZuCo preprocessing stages; interpretable latent alignment. |
| `GraphAlign-main` | Core research influence | LOSO split discipline; subject normalization; shared text space; hard negatives; subject interaction graph; safe subject adaptation; evidence that test-time adaptation cannot recover semantics absent from the base encoder. |
| `brain2qwerty-main` v1 | Adapt concepts, respect license | Convolutional neural front end; keystroke-aligned windows; sentence Transformer; character metrics; n-gram rescoring; subject/session event construction. |
| `brain2qwerty-main` v2 | Adapt concepts, respect license | Continuous Conv-Conformer encoder; character CTC; word segmentation; word-level contrastive loss; staged CTC/contrastive/LLM schedule; LoRA language head; semantic error rate. |
| `brain-to-text-working-main` | Dataset and baseline track | T15 loaders; day/session-specific input layers; GRU and RNN-T baselines; CTC; temporal and spectral augmentation; phoneme evaluation; n-gram/OPT rescoring interface. Do not copy the vendored SRILM/Wenet/Redis trees into the new repository. |
| `working_with_ZuCo_EEG_dataset-main` | Starting point for data adapter | MATLAB structure parsing; fixation extraction; word-level band-power features; ZuCo 1/2 dataset organization and sanity checks. Rewrite as tested package code rather than notebook-only logic. |
| `Brain_typing-master` | Historical baseline | Robustness to incomplete/corrupted EEG through reconstruction; recurrent-convolutional classification baseline; legacy TensorFlow code is not a production dependency. |
| `bai-64-Mind-main` | Optional classification baseline only | Inner-speech command classification API and real-time wrapper ideas. Closed data and beta weights prevent it from being a core reproducible track. |
| `DeWave-main` | Paper/result reference; implementation unavailable in this archive | Discrete EEG codex and fixation-free/raw-period motivation; define continuous EEG-to-text baselines without gold word boundaries. The added archive does not contain the claimed model source, so reproduce independently and label it as a reimplementation. |
| `EEG-To-Text-main` | Mandatory evaluation/audit reference; clean-room reuse pending license | Reproduce BrainTranslator under both teacher-forced and true `generate` evaluation; use its real-EEG versus Gaussian-noise experiments; convert its discovery into target-free generation invariance tests. Do not inherit its label-accepting `generate` signature or copy code without a confirmed license. |
| `LaBraM-main` | Foundation-encoder baseline; MIT code | VQ neural spectrum prediction; amplitude/phase reconstruction; normalized EMA codebook; 8,192-code tokenizer; masked EEG-code modeling; base/large/huge scaling references. Wrap behind the common encoder API and benchmark frozen versus fine-tuned use; audit checkpoint/data terms separately. |
| `brainmagick-main` | Research reference; CC BY-NC 4.0 code cannot enter the Apache core | Coordinate-aware channel merger, subject embeddings/layers, convolutional temporal encoder, brain-to-audio/Wav2Vec contrastive learning, negative sampling, scale rejection, and multi-dataset study abstraction. Reimplement concepts independently for the permissive core unless separately distributed as a noncommercial plugin. |
| `MEG-XL-main` | Frontier long-context MEG baseline; MIT code | BioCodec RVQ tokenizer; sensor position/orientation/type embeddings; heterogeneous sensor masks; temporal block masking; criss-cross time/channel attention; multi-dataset training; T5-space word retrieval; long-context evaluation. Keep its high-compute track optional and audit tokenizer checkpoint/data terms separately. |
| `semantic-decoding-main` | Methodological reference; clean-room reuse pending license | Separate LM proposal probabilities from a brain encoding-model likelihood during beam search; validation-fitted likelihood combination; semantic and lexical evaluation. Its fMRI timing/voxel model is not reusable as an EEG encoder, and code is not copied without confirmed terms. |
| `SQL-of-Thought-main` | Explicitly excluded | It is a text-to-SQL agent project and has no scientific or architectural role in neural decoding. Its presence is documented so agents do not accidentally import it. |

### 4.2 Paper traceability matrix

| Paper | Use |
|---|---|
| `2410.07507v1.pdf` | Historical version of Thought2Text; use to trace the original three-stage proposal and earlier evaluation claims. |
| `2410.07507v2.pdf` | Canonical Thought2Text reference; reproduce and strengthen its ablations and EEG-only evaluation. |
| `2501.06326v1.pdf` | Use its vocabulary-size, electrode-density, data-volume, error-analysis, and decoder-capacity questions as benchmark axes. |
| `2502.17480v1.pdf` | Brain2Qwerty preprint; use synchronous typing architecture, EEG/MEG comparison, CER analysis, and motor-versus-cognitive interpretation. |
| `s41593-026-02303-2.pdf` | Canonical peer-reviewed Brain2Qwerty v1 result; use its evaluation protocol and exact limitation framing. |
| `brain2qwerty_v2.pdf` | Use its continuous asynchronous decoding, character/word/sentence hierarchy, scaling results, and staged multi-objective training. |
| `EthanTrepka.pdf` | Use as evidence that a simple speech-domain n-gram can outperform a larger generic LM prior and that replacing an RNN with a Transformer is not automatically beneficial. |
| `2024.acl-long.393.pdf` | CET-MAE/E2T-PTR reference; implement its contrastive EEG-text masked autoencoding and pretrained BART decoding as a baseline, then reevaluate with target-free generation and all neural controls. |
| `2309.14030v1.pdf` through `2309.14030v4.pdf` | DeWave revision history; use the latest method description while recording changes across versions. Reproduce the vector-quantized discrete codex and raw/fixation-free input claim independently because the added repository lacks source. |
| `2405.06459v1.pdf` through `2405.06459v4.pdf` | NeuSpeech/“Are EEG-to-Text Models Working?” revision history; treat the teacher-forcing and noise-input findings as a mandatory validity audit for every generative baseline. |
| `2504.21214v1.pdf` and `2504.21214v2.pdf` | ActiveLBLM; use future spectro-temporal prediction and silent-articulation pretraining ideas, but preserve the actual 12-subject, 24-word constrained classification scope. |
| `2602.02494v1.pdf` and `2602.02494v2.pdf` | MEG-XL; use long-context multi-dataset MEG pretraining, BioCodec tokens, criss-cross attention, and word-decoding evaluation as the foundation-model expansion track. |

### 4.3 Baseline reproduction matrix introduced by the new work

| Baseline | Input contract | Output/evaluation | Required correction or comparison |
|---|---|---|---|
| BrainTranslator | word-aligned ZuCo band features | BART text | run true target-free generation; teacher-forced argmax is diagnostic only |
| E2T-PTR/CET-MAE | word-aligned EEG plus paired text during training | BART text/retrieval | rerun with unique-text/LOSO splits, zero/noise/mask controls, and label-free generation |
| DeWave | continuous/raw EEG periods | discrete codex to text | independent implementation; compare with and without fixation/word-boundary metadata |
| LaBraM | raw EEG patches, nominally 200 Hz | continuous features or VQ codes | frozen probe, partial fine-tune, from-scratch, and continuous-feature ablation |
| BrainMagick | continuous EEG/MEG plus sensor positions and audio stimulus | brain-audio retrieval/features | compare audio-target versus text-target alignment on the same split |
| MEG-XL | continuous multi-dataset MEG plus sensor geometry | masked RVQ codes and word retrieval | report compute, context length, dataset mix, frozen/linear-probe/full-fine-tune results |
| Huth semantic decoder | slow fMRI responses plus candidate text | beam candidates scored by LM and brain likelihood | port only evidence/prior factorization; do not transfer fMRI assumptions to EEG |
| ActiveLBLM | silent articulation EEG | 24-way word/semantic classification | report as constrained classification, never as open-vocabulary text generation |

### 4.4 External adjacent work

Concept2Brain is a reverse-direction model (`text/image -> synthetic EEG`) and is not part of the thought-to-text model. It should be cited only in the related-work section or a future cycle-consistency experiment. It is not an evaluation baseline for neural-to-text decoding.

## 5. Scientific scope and claims

### 5.1 Supported claims

- Decode stimulus-associated or intended text in a named experimental paradigm.
- Measure how much sentence-specific semantic information is recoverable from neural input.
- Compare modalities, channel counts, subjects, sessions, and decoder priors.
- Produce calibrated retrieval candidates or text with evidence scores.
- Evaluate adaptation to held-out subjects.

### 5.2 Unsupported claims

- Read arbitrary thoughts.
- Recover private internal monologue without task-specific training.
- Diagnose neurological or psychiatric conditions.
- Infer intent, guilt, truthfulness, or mental state.
- Operate as a clinical device.
- Generalize from healthy research participants to patients without a dedicated study.

## 6. System architecture

### 6.1 End-to-end flow

```text
Raw/preprocessed neural signal
  -> dataset-specific preprocessing
  -> modality and montage adapter
  -> continuous encoder and/or neural tokenizer
  -> subject/session adapter
  -> neural token sequence
  -> semantic bottleneck and anchor head
  -> task-specific decoding head
  -> independent neural-evidence scorer + constrained language prior
  -> text + confidence + evidence report
```

The language model must never receive the gold sentence, stimulus identifier, image class, or a lookup key at inference time.

### 6.2 Canonical sample schema

```python
@dataclass
class NeuralTextSample:
    sample_id: str
    dataset_id: str
    modality: Literal["eeg", "meg", "ecog", "intracortical"]
    paradigm: Literal["reading", "visual", "typing", "attempted_speech", "inner_speech"]
    subject_id: str
    session_id: str
    signal: FloatTensor            # [channels, time] or [events, features]
    sample_rate_hz: float
    channel_names: list[str]
    channel_positions: FloatTensor | None  # [channels, 3]
    channel_orientations: FloatTensor | None  # [channels, 3], especially MEG
    channel_types: list[str] | None        # EEG, MAG, GRAD, reference, etc.
    channel_mask: BoolTensor
    text: str
    event_onsets_s: FloatTensor | None
    event_durations_s: FloatTensor | None
    event_labels: list[str] | None
    task_labels: dict[str, Any]
    split_group: str               # unique stimulus/sentence group
    alignment_access: Literal[
        "none", "trial", "fixation", "word", "keystroke", "phoneme"
    ]
    oracle_fields: list[str]       # metadata unavailable in a deployment-like run
    provenance: dict[str, Any]
```

All adapters must return this schema. Dataset-specific fields belong in `provenance`, not in model code.

### 6.3 Preprocessing contract

Every preprocessing recipe must record:

- original file and checksum;
- dataset version and license;
- subject and session mapping;
- reference scheme;
- original and target sample rate;
- filters, notch frequencies, and filter order;
- artifact rejection and ICA decisions;
- rejected/interpolated channels;
- epoch boundaries and padding;
- normalization statistics and the split on which they were computed;
- alignment source for words, characters, phonemes, fixations, or keystrokes;
- whether sequence length, masks, event count, word count, or gold boundaries are exposed to the model;
- output checksum and schema version.

Normalization statistics must be fitted only on the training split. Test-subject statistics cannot leak into a strict LOSO experiment unless the protocol explicitly defines unsupervised test-time normalization and reports it separately.

Each artifact additionally emits an `information_access_manifest.json` recording every field visible at train, validation, and inference time. A continuous result is invalid if its batching mask or tensor length is derived from the gold word count. Fixed-duration or signal-derived chunking is required for `alignment_access="none"`.

### 6.4 Modality and montage adapter

Input forms:

- dense raw signal `[B, C, T]`;
- precomputed event features `[B, N, C, F]`;
- intracortical feature stream `[B, T, F]`.

Components:

1. per-channel robust scaling;
2. missing-channel mask;
3. coordinate-based channel embedding;
4. optional sensor-orientation and sensor-type embeddings for MEG;
5. optional Fourier-position channel merger for heterogeneous montages;
6. optional graph message passing over electrode positions;
7. dataset-specific linear or 1x1 convolution adapter;
8. temporal resampling or patchification;
9. modality token, montage token, dataset token, and sample-rate embedding.

Default EEG configuration:

- target sample rate: 250 Hz when raw signals permit;
- temporal patch: 50 ms with 25 ms stride;
- encoder width: 512;
- coordinate embedding: 64;
- graph layers: 2;
- graph neighbors: 4 to 8, configured by montage density;
- channel dropout during training: 0.05 to 0.20.

The exact filter and window settings remain dataset configs, not universal constants.

Montage adaptation must expose two interchangeable implementations:

- `graph_adapter`: named channels plus coordinate-neighbor message passing, derived from GraphAlign;
- `coordinate_merger`: learned output channels attending to Fourier-embedded sensor coordinates, derived from BrainMagick and extended with MEG-XL position/orientation/type metadata.

Both implementations consume a real-channel mask and must be invariant to padded-channel values. Coordinate normalization is performed per sensor layout using training metadata, never inferred from test signal values.

### 6.5 Shared neural encoder

Recommended default:

- convolutional stem for local denoising and downsampling;
- 8-layer Conformer;
- model dimension 512;
- 8 attention heads;
- convolution kernel 31;
- feed-forward expansion 4;
- relative temporal position encoding;
- attention and convolution masks for variable-length trials;
- output neural token sequence `[B, N, 512]`.

Required variants:

- `tiny`: 4 layers, width 256, for tests and single-GPU iteration;
- `base`: 8 layers, width 512, flagship configuration;
- `large`: 12 layers, width 768, research-only scaling experiment;
- `gru_baseline`: day/session-specific input projection plus stacked GRU;
- `channelnet_baseline`: Thought2Text-compatible visual EEG baseline.

Foundation variants behind the same `NeuralEncoder` interface:

- `labram`: temporal convolution plus channel/time embeddings and masked VQ-code prediction;
- `brainmagick_conv`: dilated convolutional sequence model with coordinate merger and subject conditioning;
- `megxl_criss_cross`: BioCodec token projection followed by alternating temporal and spatial attention;
- `continuous_conformer`: boundary-free streaming/overlap-chunk encoder for DeWave-style evaluation.

Every foundation encoder returns both `tokens: [B, N, D]` and a `TokenTiming` map from tokens back to source samples. Token timing is required for streaming, occlusion, CTC, and evidence localization.

Recommended initial wrapper defaults, subject to dataset-specific overrides:

| Wrapper | Signal/token defaults | Core trainable blocks | Required diagnostics |
|---|---|---|---|
| `labram` | resample compatible raw EEG to 200 Hz; 200-sample temporal patches; 8,192-entry, 32- or 64-dimensional VQ codebook depending checkpoint | tokenizer frozen first, then upper encoder blocks | amplitude/phase reconstruction, active codes, perplexity, mask accuracy |
| `brainmagick_conv` | continuous native windows resampled per config; coordinate Fourier merger; frozen audio features | channel merger, subject layer, dilated conv stack, projection | brain/audio retrieval, negative-pool composition, scale rejection, subject transfer |
| `megxl_criss_cross` | 250 Hz; BioCodec downsampling ratio 12; six RVQ levels with 256 bins for the provided architecture | RVQ projector, geometry embeddings, alternating time/channel transformer | per-level code accuracy, mask fraction, context length, sensor-mask invariance |
| `continuous_conformer` | fixed 2–8 s chunks with configured overlap and bounded lookahead | convolutional stem, Conformer, alignment head | latency, real-time factor, boundary-free WER/CER/retrieval, oracle gap |

Checkpoint-derived values override these defaults only through a validated config manifest. No silent resampling, patch-size, channel-order, or codebook change is permitted.

### 6.6 Subject and session adaptation

Implement three mutually comparable modes:

1. `population`: no subject-specific trainable parameters;
2. `embedding`: learned subject and session embeddings for seen participants;
3. `adapter`: small low-rank adapters conditioned on subject/session metadata.

For unseen subjects:

- population fallback;
- optional unlabeled calibration using normalization or masked reconstruction;
- optional labeled few-shot calibration at 1, 5, 10, and 30 minutes;
- no test-label use outside the explicitly reported few-shot setting.

The graph-based shared-space model is trained in the base encoder. Test-time adaptation is secondary and must not be presented as a substitute for cross-subject representation learning.

### 6.7 Neural tokenizer and semantic bottleneck

The bottleneck must prevent the text decoder from ignoring neural input.

Initial implementation:

- project neural tokens to a 512-dimensional shared EEG-text space;
- learn 8 to 32 semantic query tokens through cross-attention;
- L2-normalize the pooled neural and text embeddings;
- train symmetric InfoNCE with in-batch and memory-bank negatives;
- include hard negatives with similar vocabulary but different meaning;
- predict a sparse ordered set of semantic anchors as an auxiliary task.

Discrete-token implementation:

- LaBraM-compatible normalized-EMA VQ with amplitude/phase spectrum reconstruction;
- BioCodec-compatible residual vector quantization with configurable codebooks and bins;
- a small project-native RVQ configuration that is feasible on one GPU;
- codebook utilization and perplexity monitoring;
- masked neural-token prediction;
- ordered anchor recovery before full sentence generation.

Default ablation grid:

| Representation | Tokenizer training | Downstream use |
|---|---|---|
| continuous Conformer tokens | masked signal reconstruction | direct contrastive and seq2seq heads |
| LaBraM-style VQ spectrum codes | reconstruct normalized FFT amplitude and phase | masked-code pretraining and text alignment |
| BioCodec RVQ codes | waveform reconstruction plus RVQ commitment | criss-cross masked-code pretraining |
| soft quantized tokens | straight-through/Gumbel ablation | measure whether discreteness itself helps |

Project-fitted codebooks use only training partitions. Imported pretrained codebooks must record exactly which participants and corpora contributed to pretraining. They are labeled clean transfer only when overlap can be excluded; otherwise their results are reported as potentially contaminated transfer and cannot support a zero-shot headline claim.

Collapse alarms:

- decoder output remains unchanged under shuffled EEG;
- codebook utilization falls below a configured threshold;
- neural-text retrieval is at chance while NLG metrics remain high;
- prompt-free or zero-signal generation matches full-model performance;
- gradients into the neural encoder vanish for sustained intervals.

Any collapse alarm fails the release gate.

### 6.8 Text representations

Use separate text components for alignment and generation:

- frozen text encoder for contrastive targets;
- tokenizer and decoder for generation;
- optional reranker or language prior.

Where stimulus audio exists, add a frozen audio encoder as a third representation target. Brain-to-audio alignment is reported independently from brain-to-text alignment. It can initialize a neural encoder but may not be counted as text decoding success.

Default reproducibility configuration:

- a small permissively licensed encoder-decoder model;
- frozen text encoder during early alignment;
- LoRA or small cross-attention modules rather than full LLM fine-tuning;
- maximum context and vocabulary recorded in the checkpoint config.

Model selection must include a license review before implementation. No checkpoint with a research-only or noncommercial dependency can be labeled as unrestricted open source.

### 6.9 Decoding heads

#### Semantic reading and visual head

- neural queries cross-attend into a seq2seq decoder;
- teacher forcing during supervised training;
- scheduled neural-only conditioning and prompt dropout;
- auxiliary semantic-anchor prediction;
- beam search and constrained decoding only as reported variants;
- return token probabilities and neural evidence score.

Two inference modes are required:

- `autoregressive`: decoder generation receives neural states and neural masks only;
- `candidate_evidence`: an LM proposes candidates and a separately trained neural encoding/retrieval model scores how well each candidate explains the signal.

The public inference API must not accept `labels`, `target_ids`, gold token length, or gold decoder attention masks.

#### Typing head

- character CTC from continuous or keystroke-aligned neural tokens;
- optional sentence Transformer refinement;
- optional n-gram beam search;
- report pre-LM and post-LM CER/WER separately.

#### Attempted-speech head

- phoneme CTC and optional RNN-T;
- session/day-specific input projections;
- n-gram decoding and optional open LLM rescoring;
- report phoneme error rate before language-model decoding.

### 6.10 Language prior separation

Every generative result must report at least:

1. neural encoder plus greedy decoder;
2. neural encoder plus beam search;
3. neural encoder plus language prior;
4. language prior with zero neural input;
5. language prior with shuffled neural input.

Define:

```text
Neural Contribution = Full Model Score - Shuffled-Neural Score
Prior Contribution  = Full Model Score - Greedy-Neural Score
Grounded Gain       = Full Model Score - max(Zero, Noise, Mask, Length, Timing, LM-only Score)
```

For lower-is-better error measures, signs are reversed or the metric is converted to a bounded utility before aggregation. `Grounded Gain`, with a confidence interval, is the strict release statistic; the other quantities remain diagnostics and none substitutes for permutation testing.

For candidate-evidence decoding, use an explicit validation-tuned factorization:

```text
score(candidate, neural) =
    lambda_neural * log p(neural | candidate)
  + lambda_lm     * log p_LM(candidate)
  + lambda_len    * length_penalty(candidate)
```

`lambda_*` values are fitted on validation data only. Report the neural likelihood alone, the LM score alone, and the combined score. This imports the useful separation in HuthLab semantic decoding without importing its fMRI-specific hemodynamic model.

### 6.11 Alignment regimes and streaming contract

Every model/checkpoint declares one regime:

| Regime | Permitted inference metadata | Prohibited hidden oracle |
|---|---|---|
| `word_aligned` | recorded word/fixation events and their measured windows | gold token IDs or sentence lookup |
| `event_aligned` | measured keystroke/phoneme events | target character/phoneme identity |
| `trial_aligned` | trial start/end only | word count and internal word boundaries |
| `continuous` | causal signal samples and optional fixed-rate timestamps | future samples, sentence length, gold boundaries |

Continuous inference uses fixed-duration overlapping chunks, causal or explicitly bounded-lookahead attention, and a streaming state object. Latency, lookahead, and real-time factor are reported with accuracy. DeWave-style claims belong in `trial_aligned` or `continuous`, never silently in `word_aligned`.

### 6.12 Recommended project architecture after the expanded review

The plan contains many baselines, but the flagship should remain a controlled composition rather than an architecture pile-up:

```text
v0.1 flagship
  montage adapter: graph_adapter OR coordinate_merger (ablation chooses)
  encoder: continuous Conformer tokens
  subject adaptation: population base + optional low-rank adapter
  bottleneck: semantic queries + neural/text contrastive alignment
  evidence heads: retrieval + ordered anchors
  text head: target-free seq2seq generation
  audit: full 20-condition faithfulness suite

v0.4 research candidates
  replace continuous tokens with LaBraM VQ or BioCodec RVQ tokens
  initialize continuous encoder with BrainMagick audio alignment
  replace joint cross-attention decoding with factorized candidate evidence
  extend context with MEG-XL criss-cross attention
```

One architectural change enters the flagship only if it improves strict neural contribution, held-out-subject performance, and at least one pre-language-model metric without failing a control. Fluency alone is insufficient.

## 7. Training stages

### Stage 0: deterministic baselines and data validation

Implement before the flagship model:

- majority/mean sentence or class baseline where applicable;
- text-frequency prior;
- linear and ridge neural-to-text-embedding retrieval;
- GRU/CTC baseline for sequential paradigms;
- ChannelNet baseline for visual EEG;
- corrected BrainTranslator baseline with target-free generation;
- E2T-PTR/CET-MAE reproduction under the same split/evaluation harness;
- mask-only, length-only, event-count-only, and timing-only baselines;
- random and shuffled-label baselines.

Exit criteria:

- all splits pass leakage checks;
- chance results are empirically measured;
- evaluation metrics have unit tests;
- at least one end-to-end baseline runs from raw/preprocessed data to report.

### Stage 1: neural self-supervised pretraining

Objectives:

- masked temporal patch reconstruction;
- masked channel reconstruction;
- multi-view consistency between augmented trials;
- optional frequency-band reconstruction;
- subject/session classification adversary only if it improves held-out-subject validation.

Foundation-objective variants:

- LaBraM-style neural spectrum prediction with VQ tokenizer followed by masked code classification;
- BrainMagick-style contrastive prediction of frozen audio features from continuous brain recordings;
- BioCodec waveform reconstruction and RVQ commitment followed by MEG-XL temporal-block code prediction;
- ActiveLBLM-style future spectro-temporal prediction for silent-articulation EEG.

These variants are compute-matched where possible and always compared with a continuous non-quantized encoder.

Augmentations:

- Gaussian noise calibrated to training statistics;
- temporal jitter;
- channel dropout;
- short temporal masking;
- amplitude scaling;
- frequency masking where physiologically appropriate.

Do not use augmentations that alter label timing without updating alignment metadata.

### Stage 2: neural-text contrastive alignment

Objectives:

- symmetric InfoNCE;
- hard-negative margin loss;
- word-level alignment where timings exist;
- sentence-level alignment;
- subject-invariant shared-space regularization;
- semantic class probes for sentiment, relation, or corpus where labels exist.

Batch construction must prevent multiple copies of the same sentence from becoming false negatives. Samples sharing a stimulus receive a common group identifier.

### Stage 3: semantic-anchor decoding

Before full sentence generation, decode recoverable evidence:

- ordered keywords;
- entities;
- sentiment or relation labels;
- character or phoneme sequences for production tasks;
- retrieval candidate IDs used only for evaluation, never as hidden decoder inputs.

This stage establishes whether the signal contains usable semantic content at the chosen granularity.

### Stage 4: full text generation

Training schedule:

1. freeze neural encoder and train projector/cross-attention;
2. unfreeze upper neural layers;
3. add LoRA to text decoder if validation justifies it;
4. progressively increase prompt and teacher-forcing dropout;
5. mix full, zero-input, shuffled-input, and noise-input batches for faithfulness calibration;
6. early-stop on a composite of retrieval, semantic similarity, and neural contribution—not BLEU alone.

Evaluation at this stage calls only a separate target-free inference path. Teacher-forced validation loss can choose a checkpoint only when a target-free validation metric and faithfulness metric are also recorded.

### Stage 5: cross-subject and graph training

- strict LOSO rotation;
- graph-based montage fusion;
- subject-balanced sampler;
- per-subject validation reports;
- outlier-subject analysis;
- population versus embedding versus adapter comparison;
- optional safe test-time adaptation with seen-domain regression checks.

### Stage 6: multi-paradigm training

Only after each track works independently:

- shared encoder trunk where signal format permits;
- modality-specific stems;
- task tokens and task-specific heads;
- balanced multi-task sampler;
- gradient conflict monitoring;
- per-task checkpoint selection;
- ablation comparing shared versus independent models.

Do not merge modalities merely to increase sample count. A shared model must beat or match independent baselines without hiding degraded tasks behind an average.

### Stage 6A: fixation-free and long-context decoding

- construct trial-aligned and continuous EEG/MEG streams without word-derived padding masks;
- train fixed-chunk continuous Conformer and DeWave-style discrete-codex baselines;
- add boundary-free CTC/RNN-T or monotonic alignment where text timing is unknown;
- use MEG-XL temporal block masking for long-context pretraining;
- evaluate word retrieval at multiple context lengths before attempting long-form generation;
- report results against word-aligned oracle, mask-only, duration-only, and LM-only conditions.

Exit criteria:

- predictions are invariant to omitted or permuted target fields;
- continuous batches contain no gold-derived sequence length;
- the neural condition beats duration/mask/noise controls;
- latency and context length are included in every result.

### Stage 7: distillation and deployment

- distill `base` encoder into `tiny`;
- train 16-channel adapter;
- export fixed-shape TorchScript first;
- add ONNX or `torch.export` only after numerical parity tests;
- FP32/FP16 bundles with SHA-256 checksums;
- desktop CPU benchmark labeled as a proxy, not a phone result;
- streaming inference API for paradigms that support it.

## 8. Losses

The implementation exposes each term independently and logs raw and weighted values.

```text
L_total =
    w_ssl * L_masked_neural
  + w_recon * L_signal_or_spectrum_reconstruction
  + w_vq * L_codebook_commitment
  + w_code * L_masked_code_prediction
  + w_contrast * L_neural_text
  + w_audio * L_neural_audio
  + w_anchor * L_anchor
  + w_seq * L_sequence
  + w_ctc * L_ctc
  + w_subject * L_subject_invariance
  + w_temporal * L_temporal_consistency
  + w_cal * L_calibration
```

Rules:

- inactive heads contribute exactly zero;
- task configs state every weight;
- learned uncertainty weighting is a later ablation, not the default;
- no loss may use test-set statistics;
- each loss has a finite-value and backward-pass unit test.
- VQ/RVQ losses log active-code fraction, perplexity, dead-code count, and commitment magnitude;
- contrastive losses identify positives by stimulus group and prevent duplicate-stimulus false negatives;
- audio alignment and text alignment remain separate logged terms;
- LM likelihood never appears inside a neural-representation training target unless the experiment is explicitly marked as distillation.

## 9. Data and split implementation

### 9.1 Dataset registry

Each dataset plugin implements:

```python
class DatasetAdapter(Protocol):
    def discover(self, root: Path) -> DatasetManifest: ...
    def validate_raw(self, manifest: DatasetManifest) -> ValidationReport: ...
    def preprocess(self, manifest: DatasetManifest, config: PreprocessConfig) -> ArtifactManifest: ...
    def iter_samples(self, artifact: ArtifactManifest) -> Iterator[NeuralTextSample]: ...
    def license_info(self) -> LicenseInfo: ...
```

Initial adapters:

- `zuco_v1`;
- `zuco_v2`;
- `thought2text_visual`;
- `spanish_bcbl_eeg`;
- `spanish_bcbl_meg`;
- `t15_attempted_speech`.

Frontier/optional adapters:

- `broderick_eeg`, `brennan_eeg`, `mous_meg`, and `gwilliams_meg` for BrainMagick-style continuous brain-audio alignment;
- `armeni_meg`, `gwilliams_meg`, `libribrain_meg`, and `libribrain100_meg` for MEG-XL reproduction where data access permits;
- `active_lbm_silent_articulation` for the 24-word closed-vocabulary task;
- `huth_fmri_semantic` as an external decoder-method validation track, not a shared encoder training source.

Adapters sharing a named dataset must use one canonical ID and versioned views rather than duplicate incompatible implementations.

### 9.2 Required split protocols

- `random_legacy`: compatibility only, never headline result;
- `unique_text`: no sentence/stimulus overlap;
- `session_holdout`;
- `loso_subject`;
- `loso_subject_unique_text`;
- `cross_task`;
- `cross_dataset` where labels and paradigms permit.

The split validator must detect:

- exact duplicate text;
- normalized duplicate text;
- shared stimulus IDs;
- image-caption duplicates;
- subject/session leakage;
- preprocessing-statistic leakage;
- train/test files pointing to the same underlying record.
- overlapping continuous windows derived from the same recording across splits;
- foundation-pretraining subjects or stimuli that overlap a claimed zero-shot test set;
- test-derived codebooks, channel maps, vocabulary, or normalization statistics;
- gold-derived tensor length or attention masks in a continuous protocol.

## 10. Evaluation plan

### 10.0 Generation validity protocol

This protocol is mandatory because the NeuSpeech work demonstrates that teacher-forced token logits can produce deceptively strong text metrics even when genuine generation fails.

- `forward(batch, labels=...)` is training/validation-loss code only;
- `generate(neural, neural_mask, config)` has no labels or targets in its type signature;
- generation begins from decoder start tokens and conditions only on permitted neural metadata;
- greedy decoding is the deterministic primary result; beam and sampling are separate rows;
- metrics are computed for every batch member, not only index zero;
- decoded token IDs, stopping reason, generation config, and raw text are saved;
- a label-invariance test verifies that omitting, shuffling, or replacing target text cannot change predictions;
- teacher-forced argmax metrics, if shown, are labeled `TF diagnostic` and excluded from headline tables.

Every benchmark row uses the label:

```text
dataset / modality / paradigm / alignment regime / subject protocol /
text constraint / decoder mode / neural representation / pretraining status
```

Example: `zuco_v1/eeg/reading/word_aligned/LOSO/open_vocab/greedy/continuous/from_scratch`. This prevents an oracle-aligned or closed-vocabulary result from being visually compared as though it were fixation-free open generation.

### 10.1 Text quality

- CER and WER;
- BLEU-1 through BLEU-4;
- ROUGE-1, ROUGE-2, and ROUGE-L;
- METEOR;
- BERTScore or an explicitly versioned open semantic metric;
- semantic error rate where the required encoder is available;
- exact match for constrained tasks.

No single metric determines success.

### 10.2 Neural grounding

- EEG-to-text and text-to-EEG retrieval Top-1/5/10;
- neural-versus-shuffled performance delta;
- zero-signal and noise-signal performance;
- prompt-free performance;
- subject and session linear probes;
- gradient attribution to neural tokens;
- input occlusion across channels and time;
- anchor precision/recall;
- performance versus amount of neural data.
- mask-only, valid-length-only, duration-only, event-count-only, and timing-only performance;
- real EEG versus distribution-matched Gaussian and phase-randomized surrogate signals;
- correct alignment versus temporally shifted neural input;
- candidate ranking by neural evidence alone versus language prior alone;
- codebook and masked-token metrics for discrete foundation models.

### 10.3 Generalization

- per-subject results and macro average;
- worst-subject result;
- held-out-session result;
- strict LOSO result;
- unseen-text result;
- simultaneous unseen-subject/unseen-text result;
- reduced-channel curves;
- training-data scaling curves.
- alignment-regime transfer: word-aligned training to trial-aligned/continuous evaluation;
- frozen, linear-probe, partial-fine-tune, and full-fine-tune foundation-model results;
- context-length curves for long-context MEG;
- cross-dataset transfer with all pretraining overlap disclosed.

### 10.4 Statistical reporting

- subject-level bootstrap confidence intervals;
- stimulus-level bootstrap confidence intervals;
- paired permutation test against shuffled neural input;
- correction for multiple comparisons when reporting channel/time analyses;
- effect sizes, not only p-values;
- seed-level variability for at least three seeds on headline configurations;
- all exclusions and failed runs documented.

### 10.5 Human or LLM evaluation

Human or LLM ratings may be secondary measures of fluency and adequacy. They cannot be the only semantic evaluation. Prompts, model versions, sampling settings, and raw judgments must be released when licensing permits.

## 11. Faithfulness and anti-hallucination test suite

Every release candidate runs:

1. correct neural input;
2. shuffled neural input within subject;
3. shuffled neural input across subjects;
4. Gaussian noise matched to signal statistics;
5. all-zero input;
6. time-reversed input;
7. channel-permuted input;
8. prompt-only input;
9. text-decoder prior only;
10. mismatched stimulus and neural input;
11. random labels;
12. held-out vocabulary and paraphrase analysis.
13. valid-mask/sequence-length only;
14. fixed duration or trial-duration only;
15. event timings with signal values removed;
16. phase-randomized surrogate signal;
17. temporally shifted signal outside the plausible response window;
18. target labels omitted, permuted, and replaced to prove inference invariance;
19. gold-boundary removal for any claimed fixation-free result;
20. candidate-evidence decoder with `lambda_neural = 0` and `lambda_lm = 0` ablations.

A generative checkpoint is not publishable if meaningful metrics remain essentially unchanged across full and shuffled/zero neural input.

## 12. Repository design

```text
openthought2text/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── SECURITY.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── src/openthought2text/
│   ├── cli/
│   ├── config/
│   ├── data/
│   │   ├── schema.py
│   │   ├── registry.py
│   │   ├── splits.py
│   │   └── adapters/
│   ├── preprocessing/
│   ├── montages/
│   ├── models/
│   │   ├── stems/
│   │   ├── encoders/
│   │   ├── graphs/
│   │   ├── adapters/
│   │   ├── bottlenecks/
│   │   ├── tokenizers/
│   │   └── decoders/
│   ├── baselines/
│   │   ├── braintranslator/
│   │   ├── cet_mae/
│   │   ├── dewave/
│   │   ├── labram/
│   │   ├── brainmagick/
│   │   └── megxl/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   ├── controls/
│   └── reporting/
├── configs/
│   ├── data/
│   ├── model/
│   ├── task/
│   └── experiment/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── scientific/
│   └── fixtures/
├── benchmarks/
├── docs/
│   ├── architecture/
│   ├── datasets/
│   ├── ethics/
│   └── reproducibility/
├── model_cards/
├── dataset_cards/
└── examples/
```

### 12.1 CLI contract

```bash
ott data discover --dataset zuco_v1 --root /path/to/zuco
ott data validate --dataset zuco_v1 --root /path/to/zuco
ott data prepare --config configs/data/zuco_v1.yaml
ott splits audit --artifact artifacts/zuco_v1 --protocol loso_subject_unique_text
ott train --config configs/experiment/zuco_conformer_alignment.yaml
ott evaluate --checkpoint runs/example/best.ckpt --suite strict
ott evaluate audit-generation --checkpoint runs/example/best.ckpt
ott evaluate compare-controls --run runs/example --controls full,shuffled,zero,noise,mask,length,timing
ott report build --run runs/example
ott export --checkpoint runs/example/best.ckpt --format torchscript
```

Every command supports `--dry-run`, prints resolved paths, and writes a machine-readable manifest.

## 13. Test strategy

### Unit tests

- schema serialization and migration;
- channel-name normalization;
- montage coordinate lookup;
- mask propagation;
- variable-length collation;
- split leakage detection;
- every loss forward/backward;
- CTC length calculation;
- greedy decoding;
- CER/WER and semantic metrics;
- subject adapter selection;
- checkpoint save/load;
- deterministic seed behavior where supported.
- target-free generation signature and label-invariance;
- padded sensor values cannot affect coordinate-merger output;
- VQ/RVQ encode/decode shapes, code range, dead-code metrics, and frozen-tokenizer behavior;
- continuous chunking cannot read target length or boundaries;
- candidate evidence score decomposes exactly into saved neural, LM, and length terms.

### Integration tests

- synthetic data -> preprocessing -> train -> evaluate;
- tiny ZuCo fixture -> baseline report;
- missing-channel inference;
- resumed training parity;
- FP32/FP16 load and inference;
- CPU-only smoke test;
- multi-GPU sampler consistency when CI resources permit.
- corrected BrainTranslator tiny fixture produces identical output when target labels are changed;
- each foundation wrapper supports frozen forward, linear probe, checkpoint loading, and canonical timing metadata;
- streaming chunks match offline bounded-context output within tolerance.

### Scientific regression tests

- shuffled labels approach chance;
- zero input does not beat neural input on a synthetic learnable dataset;
- no train/test stimulus overlap;
- retrieval positives share the correct group ID;
- LM-only baseline remains separate;
- added subject adaptation cannot silently change seen-domain evaluation protocol;
- reported metric tables can be regenerated from saved predictions.
- a deliberately teacher-forced evaluator is rejected by the audit suite;
- a model exploiting only valid sequence length is detected by length/mask controls;
- a test-pretrained tokenizer or overlapping continuous window is detected as leakage;
- LM-only candidate ranking cannot be mislabeled as neural decoding.

## 14. Configuration and experiment tracking

Each run records:

- git commit and dirty status;
- resolved config;
- Python and dependency versions;
- CUDA/cuDNN versions;
- hardware and precision;
- dataset artifact checksums;
- split manifest checksum;
- random seeds;
- parameter counts by component;
- training time and peak memory;
- best-checkpoint selection rule;
- predictions and per-sample metrics.

Experiment tracking must work locally without a commercial service. External tracking integrations are optional plugins.

## 15. Detailed 28-week implementation schedule

Weeks 1–20 produce the credible ZuCo-first v0.1 release and existing production-task extensions. Weeks 21–28 are a frontier expansion for the newly added foundation-model and continuous-decoding work. The schedule assumes several agents can work in parallel. A single developer should treat it as a dependency-ordered roadmap rather than a literal calendar.

### Week 1: governance, scope, and repository foundation

Implementation:

- create the new repository rather than editing the archived sources in place;
- add Apache-2.0 candidate license pending dependency audit;
- add contributor, security, ethics, and citation documents;
- create `src` package layout and CLI skeleton;
- configure formatting, linting, type checking, tests, and CI;
- add synthetic neural-text fixture generator;
- write decision records for supported paradigms and claims.
- add a research traceability registry covering every local repository and every paper version;
- add ADRs separating word-aligned, trial-aligned, event-aligned, and continuous claims.

Deliverables:

- installable empty package;
- passing CI on Linux CPU;
- `ott --help`;
- project charter and terminology guide.

Exit criteria:

- no upstream code has been copied without provenance and license review;
- one-command development setup works;
- synthetic smoke test passes.

### Week 2: canonical schema, registry, manifests, and split engine

Implementation:

- implement `NeuralTextSample` and versioned artifact schema;
- implement dataset registry and adapter protocol;
- implement provenance and checksum manifests;
- implement variable-length collation;
- implement split protocols and leakage auditor;
- add normalized-text duplicate detection;
- add subject/session/stimulus group constraints.
- implement `alignment_access`, `oracle_fields`, sensor orientation/type, and token-timing metadata;
- generate an information-access manifest for every split and inference mode.

Deliverables:

- schema documentation;
- synthetic adapter;
- split manifest format;
- leakage report CLI.

Exit criteria:

- deliberate leakage fixtures are detected;
- schema round-trips without loss;
- train-only normalization is enforced by API.

### Week 3: ZuCo discovery and raw validation

Implementation:

- port MATLAB structure discovery from the ZuCo workspace into tested code;
- detect ZuCo versions, tasks, subjects, channels, and missing fields;
- validate sentence and word counts;
- validate fixation and EEG alignment;
- generate a data-quality report before preprocessing;
- document manual download and directory layout without redistributing participant data.

Deliverables:

- `zuco_v1` adapter discovery layer;
- raw-data validation report;
- dataset card and license record.

Exit criteria:

- all locally available subjects/tasks are inventoried;
- malformed and partial downloads fail with actionable messages;
- no participant data enters source control.

### Week 4: ZuCo preprocessing and reproducible artifacts

Implementation:

- support the established precomputed band-power path first;
- add raw/ERP path only when input records permit it;
- extract sentence and word event sequences;
- preserve fixation timing and task labels;
- emit both word-aligned examples and fixed-duration continuous chunks;
- ensure continuous chunk lengths and masks come only from signal time, never sentence word count;
- apply training-only normalization;
- write sharded tensor artifacts and indexes;
- compare a sample of outputs against existing notebooks/scripts.

Deliverables:

- versioned ZuCo artifact;
- preprocessing config;
- numerical parity report;
- visualization notebook for sanity checks.

Exit criteria:

- deterministic reruns produce matching manifests;
- sample counts reconcile with source records;
- channel/time/feature shapes pass validation.

### Week 5: evaluation harness and classical baselines

Implementation:

- CER, WER, BLEU, ROUGE, METEOR, and retrieval metrics;
- subject/stimulus bootstrap utilities;
- permutation testing;
- mean/frequency baselines;
- ridge regression from EEG features to frozen text embeddings;
- nearest-neighbor retrieval;
- report generator with per-subject tables.
- implement a corrected BrainTranslator evaluator with a target-free `generate` signature;
- implement teacher-forcing detection and label-invariance tests;
- add mask-only, length-only, duration-only, timing-only, Gaussian, and phase-surrogate baselines.

Deliverables:

- first benchmark report;
- empirical chance distributions;
- machine-readable predictions.

Exit criteria:

- metric tests use known examples;
- reports regenerate from saved predictions;
- unique-text and LOSO protocols both run.
- a deliberately label-conditioned evaluator fails the generation audit;
- predictions are identical when unused target fixtures are permuted.

### Week 6: neural stems, montage adapter, and encoder baselines

Implementation:

- dense EEG feature stem;
- raw-signal convolutional stem;
- channel mask and coordinate embeddings;
- 1x1 channel adapter inspired by the 16-channel project;
- GRU and ChannelNet baselines;
- tiny/base Conformer skeleton;
- continuous boundary-free Conformer skeleton with `TokenTiming` output;
- Fourier coordinate-merger baseline with sensor masks;
- shape, padding, and gradient tests.

Deliverables:

- common `NeuralEncoderOutput` API;
- benchmarked tiny forward pass;
- baseline training configs.

Exit criteria:

- all encoders consume the canonical batch;
- missing-channel tests pass;
- tiny models overfit a synthetic dataset.

### Week 7: self-supervised neural pretraining

Implementation:

- temporal and channel masking;
- reconstruction decoder;
- augmentation library with alignment-aware transforms;
- multi-view consistency objective;
- training loop, checkpointing, resumption, and mixed precision;
- representation probes by subject, task, and sentence.
- add project-native small VQ/RVQ tokenizer interfaces and codebook health metrics;
- reserve LaBraM/BioCodec-compatible configs without requiring their large checkpoints.

Deliverables:

- pretrained EEG encoder checkpoint;
- probe report;
- augmentation ablation.

Exit criteria:

- reconstruction beats mean baseline;
- resume-from-checkpoint is numerically sane;
- no validation statistics enter training transforms.

### Week 8: EEG-text contrastive alignment

Implementation:

- frozen text encoder interface;
- word- and sentence-level text embeddings;
- symmetric InfoNCE;
- stimulus-aware batch sampler;
- memory bank and hard negatives;
- retrieval evaluator;
- prompt-free/noise-input controls.
- implement a tiny CET-MAE/E2T-PTR-style contrastive masked-autoencoding baseline;
- add optional BrainMagick-style audio-feature targets where aligned audio exists.

Deliverables:

- aligned encoder checkpoint;
- in-subject and unique-text retrieval results;
- neural contribution report.

Exit criteria:

- full neural input significantly beats shuffled input;
- full neural input significantly beats mask-only, length-only, and timing-only inputs;
- false negatives sharing a stimulus are masked;
- retrieval gains reproduce across at least three seeds for the tiny model.

### Week 9: semantic query tokens and anchor decoder

Implementation:

- learned semantic query tokens;
- cross-attention pooling;
- keyword/entity/label heads where annotations permit;
- ordered-anchor decoder;
- anchor vocabulary construction from training data only;
- sparsity and calibration losses.

Deliverables:

- anchor predictions with confidence;
- anchor precision/recall report;
- comparison against direct retrieval.

Exit criteria:

- anchor performance beats frequency baseline;
- vocabulary construction has no test leakage;
- confidence calibration is measured.

### Week 10: seq2seq text decoder and staged training

Implementation:

- neural-prefix/cross-attention interface to text decoder;
- frozen decoder baseline;
- projector training;
- LoRA configuration;
- teacher-forcing and prompt dropout schedule;
- greedy and beam decoding;
- separate LM-only control.
- add candidate-evidence inference that stores separate neural-likelihood and LM scores;
- prohibit labels, target masks, gold lengths, and stimulus keys from the inference API.

Deliverables:

- first complete EEG-to-text checkpoint;
- pre-LM/post-LM metric table;
- saved token probabilities and evidence scores.

Exit criteria:

- full model beats zero and shuffled neural input on semantic metrics;
- generation does not depend on a stimulus lookup key;
- text decoder license is documented.

### Week 11: faithfulness, collapse detection, and error taxonomy

Implementation:

- all twenty expanded anti-hallucination and oracle-access conditions;
- automatic collapse alarms;
- error categories for omission, semantic substitution, hallucination, repetition, syntax-only recovery, and prior-dominated output;
- per-subject and per-sentence error explorer;
- channel/time occlusion.
- automatic teacher-forcing/label-dependence audit;
- word-boundary removal and fixed-duration continuous control;
- control leaderboard for full, shuffled, zero, noise, mask, length, timing, and LM-only conditions.

Deliverables:

- faithfulness dashboard/report;
- failure case gallery;
- release-blocking thresholds.

Exit criteria:

- every control runs from one command;
- metric differences have confidence intervals;
- any failed faithfulness gate is documented instead of hidden.

### Week 12: cross-subject graph encoder and LOSO

Implementation:

- electrode graph construction;
- graph fusion in base encoder;
- compare graph fusion with the coordinate-aware channel merger;
- add MEG-ready sensor orientation/type fields and padded-layout invariance tests;
- subject-balanced sampler;
- population, embedding, and low-rank adapter modes;
- full LOSO rotation;
- outlier-subject diagnostics;
- optional test-time adaptation behind an experimental flag.

Deliverables:

- strict LOSO benchmark;
- graph/no-graph ablation;
- subject adaptation report.

Exit criteria:

- graph model does not improve unseen subjects by silently using test labels;
- seen-domain regressions are reported;
- worst-subject result is included.

### Week 13: Brain2Qwerty dataset adapter and synchronous typing baseline

Implementation:

- SpanishBCBL discovery and event schema;
- EEG and MEG preprocessing kept separate;
- keystroke-aligned 500 ms windows as a configurable baseline;
- character vocabulary and labels;
- convolution plus sentence-context model;
- CER/WER extraction;
- n-gram decoding interface.

Deliverables:

- typed-production adapter;
- small reproducible subset recipe;
- pre-LM and post-LM typing results.

Exit criteria:

- EEG and MEG reports cannot be accidentally combined;
- event alignment sanity plots pass review;
- synchronous results declare keystroke timing as an event oracle;
- license and storage requirements are documented.

### Week 14: continuous typing encoder and hierarchical losses

Implementation:

- asynchronous continuous Conv-Conformer input;
- character CTC;
- CTC space segmentation;
- word-level contrastive alignment;
- staged CTC -> contrastive -> LLM training schedule;
- character, word, and semantic error metrics.
- apply the shared fixed-chunk/streaming contract and ensure no gold sentence length enters CTC batching.

Deliverables:

- continuous typing checkpoint;
- synchronous/asynchronous comparison;
- data-scaling curve where compute permits.

Exit criteria:

- CTC-only performance is reported;
- word loss has backward and segmentation tests;
- LM improvement is separated from neural improvement.

### Week 15: T15 attempted-speech adapter and GRU/CTC reproduction

Implementation:

- HDF5 discovery and block/session metadata;
- day-specific linear input layers;
- GRU baseline;
- neural augmentations from the workspace baseline;
- phoneme CTC;
- validation PER;
- file-based decoder interface that does not require Redis for basic tests.

Deliverables:

- T15 adapter;
- reproduced or carefully qualified baseline;
- phoneme logits and PER report.

Exit criteria:

- training and evaluation work without the full language model;
- day/session mapping is deterministic;
- test labels are never expected when unavailable.

### Week 16: attempted-speech language decoding and rescoring

Implementation:

- lexicon and n-gram adapter;
- optional external SRILM/Kaldi integration without vendoring;
- beam-search decoding;
- optional open LM rescoring;
- spoken-domain versus generic-language prior comparison;
- latency and memory measurements.

Deliverables:

- PER/CER/WER decomposition;
- language-prior comparison inspired by the Trepka report;
- documented external decoder setup.

Exit criteria:

- simple n-gram baseline is included;
- decoder can be disabled cleanly;
- no restrictive third-party source is copied into the repository.

### Week 17: unified multi-paradigm API and ablations

Implementation:

- modality-specific stems under one API;
- continuous features, VQ codes, and RVQ codes under one neural-token API;
- task tokens and task heads;
- balanced multi-task sampler;
- gradient norm/conflict logging;
- shared-versus-independent encoder experiments;
- checkpoint compatibility rules.

Deliverables:

- one inference API across supported tracks;
- multi-task ablation report;
- extension tutorial.
- foundation-wrapper interface documentation for LaBraM, BrainMagick, and MEG-XL.

Exit criteria:

- each independent task retains a standalone path;
- shared training does not hide task regressions;
- unsupported modality/task combinations fail clearly.

### Week 18: reduced-channel model, distillation, and export

Implementation:

- named 16-channel montage selection;
- 16-to-dense adapter;
- channel-selection and channel-dropout experiments;
- base-to-tiny distillation;
- FP16 compression;
- TorchScript export;
- checksums and offline load tests;
- CPU benchmark.

Deliverables:

- compact encoder bundle;
- channel-count curve;
- desktop latency report;
- export parity report.

Exit criteria:

- exported and eager outputs match within tolerance;
- benchmark hardware is reported;
- reduced-channel limitations are explicit.

### Week 19: documentation, demo, model cards, and ethics review

Implementation:

- architecture and data-flow documentation;
- dataset cards and model cards;
- reproducibility walkthrough;
- local demo with signal view, output text, anchors, and confidence;
- accessibility and privacy review;
- responsible-use and prohibited-use language;
- contributor onboarding and “add a dataset” guide.

Deliverables:

- documentation site;
- local demo;
- complete cards for release checkpoints;
- ethics checklist.

Exit criteria:

- demo never labels output as ground truth thought;
- all licenses and data access conditions are visible;
- a new contributor can run the synthetic example without private data.

### Week 20: reproducibility freeze and public release

Implementation:

- clean-environment reproduction;
- final multi-seed benchmark runs;
- regenerate all tables from artifacts;
- dependency and license audit;
- security review of loading and demo paths;
- tag release candidate;
- write technical report and limitations;
- archive configs, manifests, checkpoints, and prediction files.

Deliverables:

- `v0.1.0` package and source release;
- pretrained checkpoint where licensing permits;
- benchmark report;
- technical manuscript/preprint draft;
- release notes and known-issues list.

Exit criteria:

- all release gates below pass;
- clean-machine instructions are verified;
- headline claims match the actual experimental protocol;
- exact artifacts behind every reported number are retained.

### Week 21: corrected literature baseline suite

Implementation:

- package the corrected BrainTranslator reproduction;
- implement CET-MAE/E2T-PTR with multi-stream contrastive masked autoencoding;
- implement a clean-room DeWave-style VQ codex baseline from the paper because source is absent;
- standardize preprocessing, splits, decoder model, and inference settings across baselines;
- run teacher-forced diagnostics only in a separately labeled report;
- run all target-free, noise, mask, length, timing, and shuffled controls.

Deliverables:

- one command comparing BrainTranslator, CET-MAE/E2T-PTR, DeWave-style, and the project baseline;
- architecture/provenance cards for each reimplementation;
- corrected reproduction table with confidence intervals.

Exit criteria:

- no generation code path accepts target IDs;
- every prediction file passes the label-invariance audit;
- deviations from papers and unavailable upstream source are documented.

### Week 22: LaBraM tokenizer and encoder integration

Implementation:

- implement the LaBraM input transform at its expected sampling and patch scale;
- wrap VQNSP token extraction and masked EEG modeling behind `NeuralEncoder`;
- log FFT amplitude/phase reconstruction, normalized-EMA VQ statistics, active codes, and perplexity;
- support local checkpoint import with key and license validation;
- add frozen-feature, linear-probe, upper-layer fine-tune, full-fine-tune, and from-scratch configs;
- compare discrete codes with pre-quantization continuous features.

Deliverables:

- LaBraM adapter and tiny synthetic tests;
- ZuCo representation/retrieval comparison;
- memory, throughput, and codebook-health report.

Exit criteria:

- tokenizer is frozen when configured and receives no gradients;
- newly fitted codebooks use only permitted training data, while imported codebooks carry a complete pretraining manifest and overlap label;
- checkpoint/data license limitations are visible in the model card;
- quantization must add value beyond a compute-matched continuous baseline to advance.

### Week 23: BrainMagick continuous brain-audio pretraining

Implementation:

- implement study/dataset abstraction for continuous EEG/MEG recordings;
- implement coordinate Fourier merger, channel dropout, subject layers/embeddings, and scale rejection;
- align neural features to a frozen audio representation with CLIP-style negatives;
- make duplicate stimulus/audio segments group-aware in negative sampling;
- add brain-audio retrieval and neural-to-audio-feature regression metrics;
- test whether audio pretraining improves text alignment after controlling for data volume.

Deliverables:

- BrainMagick-compatible encoder wrapper;
- brain-audio retrieval benchmark on any legally available corpus;
- audio-pretrained versus text-only versus from-scratch ablation.

Exit criteria:

- stimulus duplicates are never treated as unqualified negatives;
- padded or missing channels cannot affect output;
- text-decoding gains survive shuffled-neural and data-volume-matched controls.

### Week 24: fixation-free continuous EEG-to-text

Implementation:

- build trial-aligned and fixed-duration ZuCo views without word/fixation inputs;
- train continuous Conformer and DeWave-style discrete-codex encoders;
- add monotonic alignment, CTC, or RNN-T experiments before unrestricted seq2seq generation;
- evaluate oracle word-aligned, trial-aligned, and continuous regimes in separate tables;
- test temporal shifts, boundary removal, duration-only input, and phase-randomized signal;
- benchmark streaming lookahead and real-time factor.

Deliverables:

- first fixation-free baseline;
- oracle-gap report quantifying the value of word/fixation boundaries;
- streaming inference demonstration on recorded data.

Exit criteria:

- tensor shapes and masks do not expose word count;
- target fields can be deleted without changing predictions;
- full signal beats duration, mask, timing, and surrogate-signal controls.

### Week 25: BioCodec and MEG-XL tokenizer integration

Implementation:

- wrap the BioCodec SEANet encoder/decoder and residual vector quantizer;
- validate per-channel encode/decode, downsampling ratio, code range, and waveform reconstruction;
- freeze the tokenizer for masked-code pretraining;
- implement sensor position, orientation, type, and real-channel mask embeddings;
- implement criss-cross alternating time/channel attention;
- implement temporal-block masking across real sensors only.

Deliverables:

- BioCodec wrapper and reconstruction report;
- MEG-XL tiny/base model configs;
- heterogeneous-layout and padding-invariance tests.

Exit criteria:

- frozen-tokenizer behavior and checkpoint import are deterministic;
- masked loss ignores padding and unmasked targets as configured;
- sensor geometry is traceable to dataset manifests;
- a one-GPU tiny configuration completes an end-to-end smoke run.

### Week 26: long-context multi-dataset MEG evaluation

Implementation:

- add permitted Armeni, Gwilliams, LibriBrain, and LibriBrain100 views;
- train with recording-aware sampling and no overlapping-window leakage;
- evaluate word retrieval using fixed frozen text embeddings;
- report top-k and balanced accuracy at multiple vocabulary and context sizes;
- compare frozen encoder, linear probe, partial fine-tune, and full fine-tune;
- disclose pretraining corpus/subject overlap and actual GPU memory/throughput.

Deliverables:

- MEG-XL reproduction or qualified partial reproduction;
- context-length, data-scaling, and cross-dataset transfer curves;
- compute and resource appendix.

Exit criteria:

- train/validation/test windows cannot overlap the same recording interval;
- balanced metrics accompany frequency-weighted metrics;
- high-compute results are not required to reproduce the core v0.1 package.

### Week 27: ActiveLBLM silent-articulation track

Implementation:

- add the 12-subject, 24-word silent-articulation schema if data access permits;
- implement future spectro-temporal prediction pretraining;
- implement word-identity and semantic-category classifiers;
- compare population, LOSO, and few-shot subject adaptation;
- add rest/no-articulation and motor/artifact controls where metadata permits;
- keep the API and report distinct from sentence-generation tracks.

Deliverables:

- constrained silent-articulation benchmark;
- pretraining and cross-subject ablations;
- task-specific model and dataset cards.

Exit criteria:

- outputs are described as 24-way classification;
- chance, class balance, per-subject, and worst-subject metrics are reported;
- no sentence or open-vocabulary claim is derived from this track.

### Week 28: evidence-factorized decoder and frontier release

Implementation:

- implement LM candidate proposal plus a separately trained neural-evidence scorer;
- fit neural/LM/length weights on validation data and freeze them before test;
- compare autoregressive cross-attention with candidate-evidence decoding;
- store per-candidate LM score, neural score, combined score, and rank;
- run `lambda_neural=0`, `lambda_lm=0`, shuffled-neural, and candidate-set controls;
- consolidate all new baseline, foundation, continuous, and constrained-task results.

Deliverables:

- evidence-factorized decoder API inspired by the Huth method without fMRI-specific assumptions;
- frontier benchmark report and updated architecture decision record;
- `v0.4.0-research` release candidate where licensing permits.

Exit criteria:

- test data never tunes score weights or candidate vocabulary;
- combined decoding demonstrably uses neural evidence;
- every task is labeled by modality, paradigm, alignment regime, vocabulary constraint, and subject protocol;
- failed or inconclusive results remain in the report as negative evidence.

## 16. Parallel agent work packages

### Agent A: platform and infrastructure

Owns:

- repository scaffold;
- config and CLI;
- CI, packaging, manifests, checkpoints;
- local experiment tracking;
- export and release automation.

Must not define scientific splits or metrics without review from Agent C.

### Agent B: data and preprocessing

Owns:

- canonical schema;
- dataset registry;
- ZuCo, SpanishBCBL, T15, optional BrainMagick/MEG-XL dataset views, and ActiveLBLM adapter;
- preprocessing parity;
- continuous-window construction and information-access manifests;
- dataset cards and provenance.

Must deliver validated artifacts before model agents use a dataset.

### Agent C: evaluation and scientific controls

Owns:

- leakage-safe split engine;
- metrics and statistics;
- chance and faithfulness controls;
- target-free generation audit and teacher-forcing detection;
- mask/length/timing/oracle controls;
- foundation-pretraining and overlapping-window leakage audits;
- reports and regression tests;
- error taxonomy.

This workstream has veto authority over headline results that fail leakage or faithfulness gates.

### Agent D: neural encoders and subject generalization

Owns:

- signal/montage stems;
- GRU, ChannelNet, and Conformer encoders;
- graph model;
- coordinate-aware channel merger;
- subject/session adapters;
- continuous Conformer, LaBraM, BrainMagick, BioCodec, and MEG-XL wrappers;
- self-supervised, VQ/RVQ, audio-alignment, and long-context objectives.

Depends on the schema from Agent B and synthetic fixtures from Agent A.

### Agent E: neural-text alignment and language decoding

Owns:

- text encoder interface;
- contrastive learning;
- semantic queries and anchors;
- seq2seq/CTC/RNN-T heads;
- language-prior separation;
- candidate-evidence factorization;
- fixation-free alignment/streaming heads;
- generation configs.

Must integrate Agent C’s zero/shuffled/noise controls during development, not at the end.

### Agent F: documentation, demo, and release research

Owns:

- architecture docs;
- model and dataset cards;
- demo;
- related-work traceability;
- reproducibility tutorial;
- release manuscript and ethics checklist.

Starts in Week 1 and documents decisions continuously.

### Dependency order

```text
Agent A scaffold ─┬─> Agent B schema/data ─┬─> Agent D encoders ─┐
                  │                        └─> Agent C splits     ├─> Agent E decoders
                  └─> synthetic fixtures ───────────────────────┘

Agent C controls/evaluation reviews every model milestone.
Agent F documents every accepted interface and release artifact.
```

## 17. Release gates

### Gate A: data integrity

- artifact manifests and checksums exist;
- counts reconcile with source data;
- no data leakage is detected;
- no participant data is committed;
- license and access requirements are documented.

### Gate B: baseline integrity

- empirical chance is measured;
- classical and neural baselines run end to end;
- metrics are unit-tested;
- predictions are saved for audit.
- BrainTranslator and other seq2seq baselines use target-free generation;
- teacher-forced diagnostics are clearly separated;
- mask, length, duration, timing, and LM-only baselines are present.

### Gate B2: inference integrity

- inference APIs cannot accept labels or target IDs;
- predictions are invariant to omitted, permuted, or replaced target fixtures;
- continuous inference receives no gold-derived word count, internal boundaries, or target mask;
- all decoder and evidence-combination settings are saved;
- every headline row declares its alignment regime.

### Gate C: neural grounding

- correct neural input beats shuffled and zero input;
- retrieval is above chance under unique-text splits;
- neural contribution is statistically supported;
- the model does not rely on hidden stimulus identifiers.
- correct neural input beats mask/length/timing and surrogate-signal controls;
- candidate-evidence gains disappear or degrade appropriately when the neural term is removed;
- foundation checkpoints disclose all pretraining overlap.

### Gate D: generalization

- per-subject and worst-subject results are present;
- LOSO is complete or clearly marked partial;
- seen/unseen subject protocols are separated;
- adaptation never uses unreported test labels.

### Gate E: reproducibility

- clean environment reproduction succeeds;
- configs and split manifests are archived;
- at least three seeds support headline results;
- tables regenerate from saved artifacts.

### Gate F: responsible release

- model and dataset cards are complete;
- limitations are prominent;
- no arbitrary-thought or clinical claims appear;
- third-party license audit passes;
- demo communicates uncertainty and paradigm constraints.

## 18. Compute profiles

### Development profile

- CPU or one modest GPU;
- synthetic fixtures or tiny subject subset;
- tiny encoder;
- no large language model;
- intended for unit, integration, and API work.

### Reproducible baseline profile

- one 16 to 24 GB GPU;
- frozen small text model;
- base encoder with gradient accumulation;
- subject/task subset or full precomputed ZuCo features.

### Research profile

- multiple GPUs for LOSO sweeps and multi-seed experiments;
- full raw-signal or large MEG datasets;
- base/large encoders;
- all controls and scaling experiments.

### Frontier foundation profile

- LaBraM full pretraining is treated as a multi-GPU research reproduction, not a routine contributor requirement;
- BrainMagick multi-study training requires dataset-specific storage and preprocessing budgets;
- MEG-XL full/long-context runs may require high-memory accelerators, while tiny and linear-probe configs must remain runnable on one GPU;
- foundation experiments report total accelerator-hours, peak memory, context length, tokenizer cost, and checkpoint size;
- no high-compute checkpoint becomes mandatory for tests, preprocessing, or the v0.1 benchmark.

Every published experiment reports actual compute rather than only a recommended profile.

## 19. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Small paired datasets | Memorization and unstable generation | Frozen text targets, unique-text splits, retrieval first, multi-seed results. |
| Language-model dominance | Fluent but neural-ungrounded text | Zero/shuffled/noise controls, prompt dropout, neural contribution metric, anchor stage. |
| Subject shift | Near-zero unseen-subject performance | Graph base, shared latent space, balanced sampling, LOSO, calibrated adapters. |
| Eye/motor confounds | Model decodes behavior rather than language | Exact paradigm claims, artifact controls, channel/time occlusion, timing-only baselines. |
| False negatives | Contrastive training punishes shared stimuli | Stimulus-group-aware sampler and masks. |
| Incompatible modalities | Multi-task model degrades silently | Modality stems, independent baselines, per-task gates. |
| Licensing | Release cannot be genuinely open | Independent implementation, dependency audit, separate weights/data licenses. |
| Dataset availability | Reproduction blocked | ZuCo-first release, synthetic fixtures, optional adapters for gated/embargoed data. |
| Overstated claims | Scientific and ethical harm | Terminology guide, model cards, ethics gate, exact task naming. |
| Compute cost | Community cannot reproduce | Tiny/base configs, frozen small text model, precomputed feature path, published resource use. |
| Teacher-forcing contamination | Gold text masquerades as generated output | Separate training and inference APIs, label-invariance tests, saved token traces, Gate B2. |
| Structural side-channel leakage | Word count, mask, duration, or event timing drives output | Information-access manifests and mask/length/timing/duration-only baselines. |
| Continuous-window leakage | Neighboring windows from one recording cross splits | Recording-interval groups and overlap-aware split audit. |
| Pretraining overlap | Foundation model has seen test subjects or stimuli | Pretraining manifests, overlap scanner, clean/contaminated result separation. |
| Codebook collapse | Discrete tokens carry little information | Active-code/perplexity alarms, dead-code tracking, continuous-feature ablation. |
| Misleading task equivalence | 24-way silent articulation is presented as sentence decoding | Task taxonomy, separate leaderboards, vocabulary/alignment labels on every result. |
| Foundation checkpoint licensing | Code is open but weights/data are not redistributable | Optional adapters, independent tiny configs, model-card license fields, no bundled restricted weights. |
| Long-context compute inflation | Improvements reflect scale or more data rather than method | Compute/data-matched ablations and frozen/linear/full-fine-tune comparisons. |

## 20. Definition of done for the first public release

The first release is complete only when:

- a user can install the package and run a synthetic example without external data;
- an authorized ZuCo user can prepare data with documented commands;
- split manifests prove unique-text and LOSO separation;
- at least one classical and one neural model train end to end;
- the flagship model produces text, anchors, confidence, and saved evidence metadata;
- correct neural input significantly outperforms shuffled and zero input;
- correct neural input significantly outperforms mask-only, length-only, timing-only, and surrogate-signal input;
- every generative result passes target-free generation and label-invariance audits;
- every result declares its alignment access and oracle fields;
- per-subject, worst-subject, and aggregate results are published;
- every result table is generated from saved predictions;
- tests, CI, model card, dataset card, ethics statement, and license audit pass;
- the project describes itself as constrained neural-activity-to-text decoding, not unrestricted mind reading.

## 21. Immediate pre-build decisions

Before agents write production code, the maintainers should record decisions for:

1. final repository and package name;
2. initial code license;
3. default text encoder and decoder, including their licenses;
4. exact ZuCo artifact type for the first milestone: precomputed band-power features, raw ERP epochs, or both;
5. available compute and storage;
6. whether Release 0.1 includes only reading EEG or also the visual EEG track;
7. public artifact hosting for checkpoints and benchmark predictions;
8. whether the first manuscript targets a methods benchmark, a model contribution, or both;
9. whether the first continuous benchmark uses raw EEG, ERP epochs, or reconstructed continuous periods;
10. which foundation checkpoints can legally be redistributed and which are adapter-only;
11. whether audio-aligned datasets are locally available for BrainMagick reproduction;
12. available high-memory compute for MEG-XL and LaBraM, with a fixed stop-loss budget;
13. whether the Huth-inspired candidate-evidence decoder belongs in v0.1 experimental features or the v0.4 research release.

The recommended defaults are:

- working name: `OpenThought2Text`;
- first scientific track: ZuCo reading EEG;
- first input path: established precomputed features, followed by raw epochs;
- first output hierarchy: retrieval -> semantic anchors -> full text;
- first encoder: tiny/base Conformer with graph-ready montage adapter;
- first text model: small frozen permissively licensed encoder-decoder;
- first headline protocol: `loso_subject_unique_text`;
- first alignment label: `word_aligned` with a separately reported `continuous` experimental track;
- first generation rule: target-free greedy decoding, with beam/LM variants reported separately;
- first foundation rule: wrappers and probes are optional until they beat a compute-matched continuous encoder under strict controls;
- first license target: Apache-2.0 for original code, with dataset and weight licenses documented separately.

These defaults optimize for credible open-source reproducibility. They do not prevent later MEG, visual EEG, inner-speech, or intracortical tracks.
