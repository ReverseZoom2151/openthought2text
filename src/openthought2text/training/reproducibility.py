"""Reproducibility primitives shared by training entry points."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and Torch without silently enabling unsafe nondeterminism."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
