import torch

from openthought2text.models import GroupAwareSymmetricInfoNCE, SemanticQueryPooler


def test_semantic_query_pooler_shapes_masks_and_gradients():
    torch.manual_seed(1)
    pooler = SemanticQueryPooler(hidden_size=12, num_queries=3, num_heads=3).eval()
    features = torch.randn(2, 5, 12, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    output = pooler(features, mask)
    assert output.query_features.shape == (2, 3, 12)
    assert output.pooled.shape == (2, 12)
    assert output.attention_weights.shape == (2, 3, 3, 5)
    assert torch.all(output.attention_weights[0, :, :, 3:] == 0)
    output.pooled.square().mean().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_semantic_query_pooler_ignores_padded_values():
    torch.manual_seed(8)
    pooler = SemanticQueryPooler(hidden_size=8, num_queries=2, num_heads=2).eval()
    mask = torch.tensor([[True, True, False, False]])
    original = torch.randn(1, 4, 8)
    changed = original.clone()
    changed[:, 2:] = torch.randn(1, 2, 8) * 1_000
    expected = pooler(original, mask).pooled
    actual = pooler(changed, mask).pooled
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_group_aware_infonce_masks_same_stimulus_false_negatives_and_backpropagates():
    # Rows 0 and 1 describe the same stimulus. Cross similarity is high, so
    # leaving it in the denominator produces a larger contrastive penalty.
    neural = torch.tensor([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]], requires_grad=True)
    text = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], requires_grad=True)
    criterion = GroupAwareSymmetricInfoNCE(temperature=0.1)
    ungrouped = criterion(neural, text)
    grouped = criterion(neural, text, torch.tensor([5, 5, 9]))
    assert grouped.false_negative_mask.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
    assert torch.isneginf(grouped.logits[0, 1]) and torch.isneginf(grouped.logits[1, 0])
    assert grouped.loss < ungrouped.loss
    grouped.loss.backward()
    assert neural.grad is not None and neural.grad.abs().sum() > 0
    assert text.grad is not None and text.grad.abs().sum() > 0
