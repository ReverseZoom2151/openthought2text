"""Dependency-light model primitives for OpenThought2Text."""

from .adaptation_contracts import (
    LoRAAdapterConfig,
    LoRAAdapterContract,
    LoRAProvenance,
    LoRAScheduleConfig,
    NeuralFeatureProjector,
    NeuralProjectorConfig,
)
from .alignment import (
    ContrastiveAlignmentOutput,
    GroupAwareSymmetricInfoNCE,
    SemanticPoolingOutput,
    SemanticQueryPooler,
)
from .baselines import ChannelNetNeuralEncoder, CompactConformerNeuralEncoder, GRUNeuralEncoder
from .bottleneck import NeuralRepresentationBottleneck, NeuralRepresentationBottleneckOutput
from .candidate_ranking import (
    CandidateRankingOutput,
    CandidateRankingTrainingOutput,
    EvidenceGroundedCandidateRanker,
    MaskedCandidateRankingLoss,
)
from .channels import CoordinateChannelMerger
from .checkpoint import (
    CheckpointArchitectureCompatibility,
    checkpoint_architecture_metadata,
    validate_checkpoint_architecture,
)
from .ctc_beam_search import (
    CTCBeamHypothesis,
    CTCBeamSearchConfig,
    CTCBeamSearchOutput,
    TargetFreeCTCBeamSearch,
    ValidationFittedLanguageScorer,
)
from .decoder import DecoderGenerationConfig, DecoderTrainingOutput, TargetFreeAutoregressiveDecoder
from .distillation import (
    ReducedChannelDistillationConfig,
    ReducedChannelDistillationLoss,
    ReducedChannelDistillationOutput,
)
from .domain_adversarial import (
    CrossSubjectAdversarialOutput,
    CrossSubjectDomainAdversary,
    gradient_reverse,
)
from .encoder import ContinuousNeuralEncoder
from .factorized_scoring import (
    EvidenceFactorizedCandidateScorer,
    FactorizedCandidateScoringOutput,
    FactorizedScoringControl,
    ValidationFittedScoreWeights,
    fit_factorized_score_weights,
)
from .factory import (
    NeuralToTextModelConfig,
    architecture_fingerprint,
    build_neural_to_text_model,
    describe_model_architecture,
)
from .foundation_wrapper import (
    FoundationEncoderWrapper,
    FoundationFeatureContract,
    FoundationPretrainingProvenance,
)
from .heads import (
    CTCProductionHead,
    CTCProductionOutput,
    SemanticAnchorHead,
    SemanticAnchorOutput,
    greedy_ctc_decode,
)
from .masked_token_pretraining import (
    MaskedNeuralTokenConfig,
    MaskedNeuralTokenObjectiveOutput,
    MaskedNeuralTokenPredictionObjective,
    select_mask_positions,
)
from .model import NeuralToTextGenerationOutput, NeuralToTextModel, NeuralToTextTrainingOutput
from .montage import GraphMontageAdapter
from .multitask import (
    BalancedTaskSchedule,
    GradientConflictLog,
    MultiParadigmTaskConfig,
    MultiParadigmTaskHead,
    MultiTaskTrainingOutput,
    TaskTrackConfig,
    gradient_conflict_log,
)
from .residual_quantizer import (
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
    ResidualVectorQuantizerOutput,
)
from .self_supervision import (
    NeuralReconstructionConsistencyObjective,
    NeuralReconstructionHead,
    NeuralSelfSupervisionConfig,
    NeuralSelfSupervisionOutput,
)
from .semantic_anchor_decoder import (
    OrderedSemanticAnchorDecoder,
    OrderedSemanticAnchorDecoderConfig,
    OrderedSemanticAnchorGenerationConfig,
    OrderedSemanticAnchorGenerationOutput,
    OrderedSemanticAnchorTrainingOutput,
)
from .subject import SubjectAdapter, SubjectAdapterMode
from .tokenizer import (
    CodebookHealth,
    NeuralTokenizerConfig,
    NeuralTokenizerOutput,
    NeuralVectorQuantizer,
    codebook_health,
)
from .torchscript_export import (
    TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE,
    NeuralEncoderEvidenceTorchScriptAdapter,
    TorchScriptExportValidation,
    export_neural_encoder_evidence_torchscript,
    validate_neural_encoder_evidence_torchscript,
)
from .types import NeuralEncoderOutput, TokenTiming

__all__ = [
    "CodebookHealth",
    "CandidateRankingOutput",
    "CandidateRankingTrainingOutput",
    "ChannelNetNeuralEncoder",
    "CheckpointArchitectureCompatibility",
    "CompactConformerNeuralEncoder",
    "ContrastiveAlignmentOutput",
    "CrossSubjectAdversarialOutput",
    "CrossSubjectDomainAdversary",
    "ContinuousNeuralEncoder",
    "CoordinateChannelMerger",
    "CTCProductionHead",
    "CTCProductionOutput",
    "CTCBeamHypothesis",
    "CTCBeamSearchConfig",
    "CTCBeamSearchOutput",
    "DecoderGenerationConfig",
    "DecoderTrainingOutput",
    "EvidenceGroundedCandidateRanker",
    "EvidenceFactorizedCandidateScorer",
    "FactorizedCandidateScoringOutput",
    "FactorizedScoringControl",
    "FoundationEncoderWrapper",
    "FoundationFeatureContract",
    "FoundationPretrainingProvenance",
    "GroupAwareSymmetricInfoNCE",
    "GraphMontageAdapter",
    "BalancedTaskSchedule",
    "GradientConflictLog",
    "MultiParadigmTaskConfig",
    "MultiParadigmTaskHead",
    "MultiTaskTrainingOutput",
    "TaskTrackConfig",
    "gradient_conflict_log",
    "GRUNeuralEncoder",
    "NeuralEncoderOutput",
    "NeuralTokenizerConfig",
    "NeuralTokenizerOutput",
    "NeuralToTextGenerationOutput",
    "NeuralToTextModel",
    "NeuralToTextModelConfig",
    "NeuralToTextTrainingOutput",
    "NeuralRepresentationBottleneck",
    "NeuralRepresentationBottleneckOutput",
    "NeuralReconstructionConsistencyObjective",
    "NeuralReconstructionHead",
    "NeuralSelfSupervisionConfig",
    "NeuralSelfSupervisionOutput",
    "NeuralVectorQuantizer",
    "MaskedCandidateRankingLoss",
    "MaskedNeuralTokenConfig",
    "MaskedNeuralTokenObjectiveOutput",
    "MaskedNeuralTokenPredictionObjective",
    "ResidualVectorQuantizer",
    "ResidualVectorQuantizerConfig",
    "ResidualVectorQuantizerOutput",
    "ReducedChannelDistillationConfig",
    "ReducedChannelDistillationLoss",
    "ReducedChannelDistillationOutput",
    "SubjectAdapter",
    "SubjectAdapterMode",
    "SemanticPoolingOutput",
    "SemanticQueryPooler",
    "SemanticAnchorHead",
    "SemanticAnchorOutput",
    "TargetFreeAutoregressiveDecoder",
    "TargetFreeCTCBeamSearch",
    "TORCHSCRIPT_SCOPE_NEURAL_ENCODER_EVIDENCE",
    "TokenTiming",
    "TorchScriptExportValidation",
    "ValidationFittedScoreWeights",
    "ValidationFittedLanguageScorer",
    "architecture_fingerprint",
    "build_neural_to_text_model",
    "codebook_health",
    "checkpoint_architecture_metadata",
    "describe_model_architecture",
    "export_neural_encoder_evidence_torchscript",
    "fit_factorized_score_weights",
    "greedy_ctc_decode",
    "gradient_reverse",
    "select_mask_positions",
    "NeuralEncoderEvidenceTorchScriptAdapter",
    "validate_neural_encoder_evidence_torchscript",
    "validate_checkpoint_architecture",
    "LoRAAdapterConfig",
    "LoRAAdapterContract",
    "LoRAProvenance",
    "LoRAScheduleConfig",
    "NeuralFeatureProjector",
    "NeuralProjectorConfig",
    "OrderedSemanticAnchorDecoder",
    "OrderedSemanticAnchorDecoderConfig",
    "OrderedSemanticAnchorGenerationConfig",
    "OrderedSemanticAnchorGenerationOutput",
    "OrderedSemanticAnchorTrainingOutput",
]
