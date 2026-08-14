"""Explicit, auditable weighted-loss composition."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def compose_losses(
    losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]
) -> torch.Tensor:
    """Sum named losses, requiring an explicit weight for every active loss."""

    if not losses:
        raise ValueError("At least one loss is required")
    unknown = set(losses).difference(weights)
    if unknown:
        raise ValueError(f"Missing weights for losses: {sorted(unknown)}")
    total = next(iter(losses.values())).new_zeros(())
    for name, value in losses.items():
        if value.ndim != 0:
            raise ValueError(f"Loss {name!r} must be scalar, received shape {tuple(value.shape)}")
        total = total + float(weights[name]) * value
    return total
