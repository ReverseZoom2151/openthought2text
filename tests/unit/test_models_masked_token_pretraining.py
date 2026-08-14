import pytest
import torch

from openthought2text.models import (
    MaskedNeuralTokenConfig,
    MaskedNeuralTokenPredictionObjective,
    select_mask_positions,
)


def test_mask_selection_never_selects_padding_and_honors_minimum_per_nonempty_row():
    config = MaskedNeuralTokenConfig(mask_ratio=0.2, minimum_masked_tokens=2)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False], [False, False, False, False]])
    selected = select_mask_positions(mask, config, torch.Generator().manual_seed(5))
    assert not (selected & ~mask).any()
    assert selected.sum(dim=1).tolist() == [2, 1, 0]


def test_masked_neural_token_objective_uses_selected_positions_levels_health_and_gradients():
    torch.manual_seed(24)
    objective = MaskedNeuralTokenPredictionObjective()
    logits = torch.randn(2, 4, 2, 5, requires_grad=True)
    token_ids = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]], [[4, 3], [3, 2], [2, 1], [1, 0]]])
    token_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    selected = torch.tensor([[True, False, False, False], [False, True, True, False]])
    output = objective(logits, token_ids, token_mask, selected)
    assert output.masked_token_count.item() == 3
    assert output.mask_positions.equal(selected)
    assert len(output.per_level_loss) == 2 and len(output.per_level_health) == 2
    assert all(item.usage.shape == (5,) for item in output.per_level_health)
    output.loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert torch.all(logits.grad[~selected] == 0)


def test_masked_neural_token_objective_rejects_padding_selection_and_misaligned_or_invalid_tokens():
    objective = MaskedNeuralTokenPredictionObjective()
    logits = torch.randn(1, 2, 1, 3)
    ids = torch.tensor([[[1], [2]]])
    token_mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="padded"):
        objective(logits, ids, token_mask, torch.tensor([[False, True]]))
    with pytest.raises(ValueError, match="share"):
        objective(torch.randn(1, 2, 2, 3), ids, token_mask)
    with pytest.raises(ValueError, match="outside"):
        objective(logits, torch.tensor([[[1], [3]]]), token_mask)
