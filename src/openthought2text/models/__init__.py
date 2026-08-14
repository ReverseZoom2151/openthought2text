"""Dependency-light model primitives for OpenThought2Text."""

from .channels import CoordinateChannelMerger
from .alignment import (
    ContrastiveAlignmentOutput,
    GroupAwareSymmetricInfoNCE,
    SemanticPoolingOutput,
    SemanticQueryPooler,
)
from .decoder import DecoderGenerationConfig, DecoderTrainingOutput, TargetFreeAutoregressiveDecoder
from .encoder import ContinuousNeuralEncoder
from .heads import (
    CTCProductionHead,
    CTCProductionOutput,
    SemanticAnchorHead,
    SemanticAnchorOutput,
    greedy_ctc_decode,
)
from .model import NeuralToTextGenerationOutput, NeuralToTextModel, NeuralToTextTrainingOutput
from .montage import GraphMontageAdapter
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
    "ContrastiveAlignmentOutput",
    "ContinuousNeuralEncoder",
    "CoordinateChannelMerger",
    "CTCProductionHead",
    "CTCProductionOutput",
    "DecoderGenerationConfig",
    "DecoderTrainingOutput",
    "GroupAwareSymmetricInfoNCE",
    "GraphMontageAdapter",
    "NeuralEncoderOutput",
    "NeuralTokenizerConfig",
    "NeuralTokenizerOutput",
    "NeuralToTextGenerationOutput",
    "NeuralToTextModel",
    "NeuralToTextTrainingOutput",
    "NeuralVectorQuantizer",
    "ResidualVectorQuantizer",
    "ResidualVectorQuantizerConfig",
    "ResidualVectorQuantizerOutput",
    "SubjectAdapter",
    "SubjectAdapterMode",
    "SemanticPoolingOutput",
    "SemanticQueryPooler",
    "SemanticAnchorHead",
    "SemanticAnchorOutput",
    "TargetFreeAutoregressiveDecoder",
    "TokenTiming",
    "codebook_health",
    "greedy_ctc_decode",
]
