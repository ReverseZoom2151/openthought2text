import pytest
import torch

from openthought2text.models import CTCProductionHead, SemanticAnchorHead, greedy_ctc_decode


def test_semantic_anchor_loss_respects_position_mask_and_gradients():
    torch.manual_seed(3)
    head = SemanticAnchorHead(hidden_size=6, num_anchors=4)
    features = torch.randn(2, 4, 6, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    first = head(features, targets, mask)
    altered = targets.clone()
    altered[~mask] = torch.tensor([0, 1, 2, 3])
    second = head(features, altered, mask)
    assert first.logits.shape == (2, 4, 4)
    torch.testing.assert_close(first.loss, second.loss)
    first.loss.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_semantic_anchor_all_masked_has_safe_differentiable_zero():
    head = SemanticAnchorHead(hidden_size=3, num_anchors=2)
    features = torch.randn(1, 2, 3, requires_grad=True)
    output = head(features, torch.tensor([[1, 0]]), torch.zeros(1, 2, dtype=torch.bool))
    assert output.loss.item() == 0.0
    output.loss.backward()
    assert features.grad is not None


def test_ctc_head_shapes_lengths_and_gradients():
    torch.manual_seed(4)
    head = CTCProductionHead(hidden_size=5, vocabulary_size=6, blank_token_id=0)
    features = torch.randn(2, 5, 5, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    targets = torch.tensor([[1, 2, 0], [3, 4, 5]])
    output = head(features, mask, targets, torch.tensor([2, 3]))
    assert output.logits.shape == (2, 5, 6)
    assert output.log_probs.shape == (5, 2, 6)
    assert output.input_lengths.tolist() == [3, 5]
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_ctc_rejects_holey_mask_and_invalid_lengths():
    head = CTCProductionHead(hidden_size=2, vocabulary_size=4)
    features = torch.randn(1, 3, 2)
    with pytest.raises(ValueError, match="prefix"):
        head(features, torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="no greater"):
        head(
            features,
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[1, 2, 3, 1]]),
            torch.tensor([4]),
        )


def test_greedy_ctc_collapse_honors_blanks_repetitions_and_prefix_mask():
    # argmax paths: [1,1,0,1,2] -> [1,1,2]; second final token is masked.
    logits = torch.full((2, 5, 3), -10.0)
    logits[0, torch.arange(5), torch.tensor([1, 1, 0, 1, 2])] = 10
    logits[1, torch.arange(5), torch.tensor([2, 2, 0, 1, 1])] = 10
    decoded = greedy_ctc_decode(
        logits,
        blank_token_id=0,
        token_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool),
    )
    assert decoded == [[1, 1, 2], [2, 1]]
