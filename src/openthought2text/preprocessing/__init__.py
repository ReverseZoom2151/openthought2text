"""Signal-only preprocessing primitives with explicit alignment boundaries."""

from .continuous import ContinuousChunks, chunk_continuous_signal, robust_channel_scale

__all__ = ["ContinuousChunks", "chunk_continuous_signal", "robust_channel_scale"]
