import inspect

import pytest
import torch

from openthought2text.models import EvidenceGroundedCandidateRanker


def _ranker() -> EvidenceGroundedCandidateRanker:
    return EvidenceGroundedCandidateRanker(hidden_size=8, candidate_embedding_dim=5, num_queries=2, num_heads=2).eval()


def test_candidate_ranker_shapes_masking_evidence_and_gradients():
    torch.manual_seed(14)
    ranker = _ranker()
    neural = torch.randn(2, 4, 8, requires_grad=True)
    neural_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    ids = torch.tensor([10, 20, 30])
    candidates = torch.randn(3, 5)
    candidate_mask = torch.tensor([[True, False, True], [True, True, True]])
    output = ranker(neural, neural_mask, ids, candidates, candidate_mask)
    assert output.scores.shape == (2, 3)
    assert output.neural_evidence.shape == (2, 5)
    assert output.query_features.shape == (2, 2, 8)
    assert output.candidate_ids.tolist() == [[10, 20, 30], [10, 20, 30]]
    assert torch.isneginf(output.scores[0, 1])
    assert output.ranked_candidate_ids.shape == (2, 3)
    output.scores[torch.isfinite(output.scores)].sum().backward()
    assert neural.grad is not None and neural.grad.abs().sum() > 0


def test_candidate_ranker_is_target_free_and_ignores_masked_neural_values():
    torch.manual_seed(15)
    ranker = _ranker()
    signature = inspect.signature(ranker.forward)
    assert set(signature.parameters) == {
        "neural_features", "neural_mask", "candidate_ids", "candidate_embeddings", "candidate_mask"
    }
    mask = torch.tensor([[True, True, False]])
    original = torch.randn(1, 3, 8)
    changed = original.clone()
    changed[:, 2] = torch.randn(8) * 1000
    ids = torch.tensor([[1, 2]])
    candidates = torch.randn(1, 2, 5)
    first = ranker(original, mask, ids, candidates).scores
    second = ranker(changed, mask, ids, candidates).scores
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_candidate_ranker_validates_authorized_candidate_shapes_and_masks():
    ranker = _ranker()
    neural = torch.randn(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    candidates = torch.randn(2, 5)
    with pytest.raises(ValueError, match="nonnegative"):
        ranker(neural, mask, torch.tensor([-1, 2]), candidates)
    with pytest.raises(ValueError, match="at least one"):
        ranker(neural, mask, torch.tensor([1, 2]), candidates, torch.tensor([False, False]))
