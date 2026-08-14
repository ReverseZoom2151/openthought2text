"""Minimal local smoke workflow; it uses no participant or external dataset data."""

from __future__ import annotations

import torch

from openthought2text.losses.composite import compose_losses
from openthought2text.training.reproducibility import seed_everything


def main() -> None:
    seed_everything(7)
    prediction = torch.tensor([0.2, 0.8])
    target = torch.tensor([0.0, 1.0])
    reconstruction = (prediction - target).square().mean()
    total = compose_losses({"reconstruction": reconstruction}, {"reconstruction": 1.0})
    print({"synthetic_loss": float(total)})


if __name__ == "__main__":
    main()
