"""Signal-preserving control constructions for neural-decoding experiments.

Every control is explicit about the side information it retains.  Do not use a
control tensor as an inference input unless its name and access manifest accompany
the recorded result.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any


def zero_signal(signal: Any) -> Any:
    """Return a zero-valued tensor/array/nested sequence with the same shape."""
    if hasattr(signal, "new_zeros"):
        return signal.new_zeros(signal.shape)
    module = type(signal).__module__
    if module.startswith("numpy"):
        return signal * 0
    if isinstance(signal, tuple):
        return tuple(zero_signal(item) for item in signal)
    if isinstance(signal, list):
        return [zero_signal(item) for item in signal]
    return 0.0


def shuffle_batch(signal: Any, *, seed: int = 0, permutation: Sequence[int] | None = None) -> Any:
    """Shuffle examples along batch axis while preserving each example exactly."""
    try:
        batch_size = len(signal)
    except TypeError as error:
        raise ValueError("batch shuffling requires a batch-like first axis") from error
    if permutation is None:
        permutation = list(range(batch_size))
        random.Random(seed).shuffle(permutation)
        # An identity permutation is not a shuffled-neural control.  This matters
        # most for tiny batches, where it is surprisingly likely.
        if batch_size > 1 and list(permutation) == list(range(batch_size)):
            permutation = [*range(1, batch_size), 0]
    if sorted(permutation) != list(range(batch_size)):
        raise ValueError("permutation must contain each batch index exactly once")
    module = type(signal).__module__
    if module.startswith("torch"):
        import torch  # Optional dependency, imported only for torch inputs.

        index = torch.tensor(permutation, device=signal.device, dtype=torch.long)
        return signal.index_select(0, index)
    if module.startswith("numpy"):
        return signal[list(permutation)]
    return [deepcopy(signal[index]) for index in permutation]


def gaussian_noise_like(signal: Any, *, seed: int = 0) -> Any:
    """Distribution-matched iid Gaussian noise (global mean and standard deviation)."""
    values = [
        value
        for value in _flatten(signal)
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not values:
        raise ValueError("signal must contain finite numeric values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    rng = random.Random(seed)
    control = _map_numeric(signal, lambda _: rng.gauss(mean, std))
    return _restore_like(signal, control)


def mask_only_signal(valid_mask: Any, *, channels: int = 1) -> Any:
    """Expose padding/validity structure but no signal values.

    A [batch, time] mask becomes [batch, channels, time], so a model cannot infer
    values but can exploit exactly the mask side channel being measured.
    """
    if channels <= 0:
        raise ValueError("channels must be positive")
    if _ndim(valid_mask) != 2:
        raise ValueError("valid_mask must have shape [batch, time]")
    return [[[float(item) for item in row] for _ in range(channels)] for row in _tolist(valid_mask)]


def length_only_signal(
    valid_lengths: Sequence[int], *, channels: int = 1, max_length: int | None = None
) -> list[list[list[float]]]:
    """Expose only valid sequence length, encoded as a prefix mask."""
    lengths = [int(length) for length in valid_lengths]
    if not lengths or any(length < 0 for length in lengths):
        raise ValueError("valid_lengths must be a non-empty sequence of non-negative values")
    width = max(lengths) if max_length is None else int(max_length)
    if width < max(lengths) or channels <= 0:
        raise ValueError("max_length must cover all lengths and channels must be positive")
    mask = [[index < length for index in range(width)] for length in lengths]
    return mask_only_signal(mask, channels=channels)


def timing_only_signal(
    event_indices: Sequence[Sequence[int]],
    *,
    time_steps: int,
    channels: int = 1,
) -> list[list[list[float]]]:
    """Rasterize event timings with all observed neural values removed."""
    if time_steps <= 0 or channels <= 0:
        raise ValueError("time_steps and channels must be positive")
    result: list[list[list[float]]] = []
    for events in event_indices:
        row = [0.0] * time_steps
        for event in events:
            if not 0 <= int(event) < time_steps:
                raise ValueError("event index is outside [0, time_steps)")
            row[int(event)] = 1.0
        result.append([row.copy() for _ in range(channels)])
    return result


def phase_randomized_surrogate(signal: Any, *, seed: int = 0) -> Any:
    """Return a deterministic value-permuted surrogate without SciPy/NumPy.

    This is a conservative fallback for environments without an FFT backend.  It
    preserves the global marginal distribution while removing time-locked signal
    structure.  FFT phase-randomization belongs in a dataset-specific optional
    preprocessor and must state its axis/mask policy.
    """
    values = list(_flatten(signal))
    rng = random.Random(seed)
    rng.shuffle(values)
    iterator = iter(values)
    control = _map_numeric(signal, lambda _: next(iterator))
    return _restore_like(signal, control)


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def _tolist(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return deepcopy(value)


def _ndim(value: Any) -> int:
    if hasattr(value, "ndim"):
        return int(value.ndim)
    dimension = 0
    current = value
    while isinstance(current, (list, tuple)):
        dimension += 1
        current = current[0] if current else []
    return dimension


def _map_numeric(value: Any, mapper: Any) -> Any:
    if hasattr(value, "detach") or type(value).__module__.startswith("numpy"):
        # Converting preserves control semantics and keeps this module dependency-free.
        value = _tolist(value)
    if isinstance(value, tuple):
        return tuple(_map_numeric(item, mapper) for item in value)
    if isinstance(value, list):
        return [_map_numeric(item, mapper) for item in value]
    return mapper(value) if isinstance(value, (int, float)) else value


def _restore_like(original: Any, control: Any) -> Any:
    """Keep tensor/array controls compatible with the model input contract."""
    module = type(original).__module__
    if module.startswith("torch"):
        return original.new_tensor(control)
    if module.startswith("numpy"):
        import numpy as np  # Optional dependency, only needed for NumPy inputs.

        return np.asarray(control, dtype=original.dtype)
    return control
