import pytest
import torch
from torch import nn

from openthought2text.models import (
    FrozenTextEmbeddingInterface,
    GroupAwareHardNegativeMemoryBank,
    TextEmbeddingContract,
    TextEmbeddingProvenance,
)


class _Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(3, 2)

    def forward(self, x, m):
        return self.p(x).sum(1) / m.sum(1, keepdim=True)


def test_frozen_text_interface_contract_provenance_and_frozen_parameters():
    interface = FrozenTextEmbeddingInterface(
        _Text(), TextEmbeddingContract(3, 2), TextEmbeddingProvenance("text", "fp", "unknown")
    )
    assert not any(x.requires_grad for x in interface.encoder.parameters())
    assert interface.training_embeddings(
        torch.randn(1, 2, 3), torch.tensor([[True, False]])
    ).shape == (1, 2)
    with pytest.raises(ValueError, match="explicit"):
        TextEmbeddingProvenance("x", "y", "bad")


def test_group_bank_fifo_deterministic_and_excludes_same_stimulus_groups():
    bank = GroupAwareHardNegativeMemoryBank(3, 2)
    bank.add_training_embeddings(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), torch.tensor([1, 1, 2, 3])
    )
    assert bank.size == 3
    sample = bank.sample_hard_negatives(torch.tensor([[0.0, 1.0]]), torch.tensor([1]), 3)
    assert sample.mask.tolist() == [[True, True, False]]
    assert all(bank._groups[i].item() != 1 for i in sample.memory_indices[0, :2])
    assert sample.memory_indices[0, :2].tolist() == [1, 2]
