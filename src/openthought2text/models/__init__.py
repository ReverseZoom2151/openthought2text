"""Dependency-light model primitives for OpenThought2Text."""

from .channels import CoordinateChannelMerger
from .checkpoint import (
    CheckpointArchitectureCompatibility,
    checkpoint_architecture_metadata,
    validate_checkpoint_architecture,
)
from .alignment import (
    ContrastiveAlignmentOutput,
    GroupAwareSymmetricInfoNCE,
    SemanticPoolingOutput,
    SemanticQueryPooler,
)
from .candidate_ranking import (
    CandidateRankingOutput,
    CandidateRankingTrainingOutput,
    EvidenceGroundedCandidateRanker,
    MaskedCandidateRankingLoss,
)
from .bottleneck import NeuralRepresentationBottleneck, NeuralRepresentationBottleneckOutput
from .decoder import DecoderGenerationConfig, DecoderTrainingOutput, TargetFreeAutoregressiveDecoder
from .distillation import (
    ReducedChannelDistillationConfig,
    ReducedChannelDistillationLoss,
    ReducedChannelDistillationOutput,
)
from .encoder import ContinuousNeuralEncoder
from .factory import (
    NeuralToTextModelConfig,
    architecture_fingerprint,
    build_neural_to_text_model,
    describe_model_architecture,
)
from .heads import (
    CTCProductionHead,
    CTCProductionOutput,
    SemanticAnchorHead,
    SemanticAnchorOutput,
    greedy_ctc_decode,
)
from .model import NeuralToTextGenerationOutput, NeuralToTextModel, NeuralToTextTrainingOutput
from .montage import GraphMontageAdapter
from .masked_token_pretraining import (
    MaskedNeuralTokenConfig,
    MaskedNeuralTokenObjectiveOutput,
    MaskedNeuralTokenPredictionObjective,
    select_mask_positions,
)
from .residual_quantizer import (
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
    ResidualVectorQuantizerOutput,
)
from .subject import SubjectAdapter, SubjectAdapterMode
from .tokenizer import (
    CodebookHealth,
    NeuralTokenizerConfig,
    NeuralTokenizerOutput,
    NeuralVectorQuantizer,
    codebook_health,
)
from .types import NeuralEncoderOutput, TokenTiming

__all__ = [
    "CodebookHealth",
    "CandidateRankingOutput",
    "CandidateRankingTrainingOutput",
    "CheckpointArchitectureCompatibility",
    "ContrastiveAlignmentOutput",
    "ContinuousNeuralEncoder",
    "CoordinateChannelMerger",
    "CTCProductionHead",
    "CTCProductionOutput",
    "DecoderGenerationConfig",
    "DecoderTrainingOutput",
    "EvidenceGroundedCandidateRanker",
    "GroupAwareSymmetricInfoNCE",
    "GraphMontageAdapter",
    "NeuralEncoderOutput",
    "NeuralTokenizerConfig",
    "NeuralTokenizerOutput",
    "NeuralToTextGenerationOutput",
    "NeuralToTextModel",
    "NeuralToTextModelConfig",
    "NeuralToTextTrainingOutput",
    "NeuralRepresentationBottleneck",
    "NeuralRepresentationBottleneckOutput",
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
    "TokenTiming",
    "architecture_fingerprint",
    "build_neural_to_text_model",
    "codebook_health",
    "checkpoint_architecture_metadata",
    "describe_model_architecture",
    "greedy_ctc_decode",
    "select_mask_positions",
    "validate_checkpoint_architecture",
]
