import pytest
import torch

from openthought2text.models import (
    NeuralReconstructionConsistencyObjective,
    NeuralReconstructionHead,
    NeuralSelfSupervisionConfig,
)


def _objective():
    return NeuralReconstructionConsistencyObjective(
        NeuralReconstructionHead(hidden_size=4, reconstruction_size=3),
        NeuralSelfSupervisionConfig(reconstruction_weight=1.0, consistency_weight=0.5),
    )


def test_self_supervision_reconstructs_and_matches_only_valid_tokens_with_gradients():
    torch.manual_seed(30)
    objective = _objective()
    primary = torch.randn(2, 4, 4, requires_grad=True)
    secondary = torch.randn(2, 4, 4, requires_grad=True)
    targets = torch.randn(2, 4, 3, requires_grad=True)
    primary_mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    secondary_mask = torch.tensor([[True, False, True, False], [True, True, True, False]])
    reconstruction_mask = torch.tensor([[True, False, False, False], [True, False, True, False]])
    output = objective(
        primary, primary_mask, targets, reconstruction_mask, secondary, secondary_mask
    )
    assert output.reconstruction.shape == (2, 4, 3)
    assert output.reconstruction_token_count.item() == 3
    assert output.consistency_mask.tolist() == [
        [True, False, False, False],
        [True, False, True, False],
    ]
    assert output.consistency_token_count.item() == 3
    output.loss.backward()
    assert primary.grad is not None and primary.grad.abs().sum() > 0
    assert secondary.grad is not None and secondary.grad.abs().sum() > 0
    assert targets.grad is None


def test_self_supervision_ignores_padded_target_and_secondary_values_deterministically():
    torch.manual_seed(31)
    objective = _objective().eval()
    primary = torch.randn(1, 3, 4)
    secondary = torch.randn(1, 3, 4)
    targets = torch.randn(1, 3, 3)
    primary_mask = torch.tensor([[True, True, False]])
    secondary_mask = torch.tensor([[True, False, False]])
    first = objective(
        primary, primary_mask, targets, secondary_features=secondary, secondary_mask=secondary_mask
    )
    changed_targets = targets.clone()
    changed_targets[:, 2] = 1_000
    changed_secondary = secondary.clone()
    changed_secondary[:, 1:] = -1_000
    second = objective(
        primary,
        primary_mask,
        changed_targets,
        secondary_features=changed_secondary,
        secondary_mask=secondary_mask,
    )
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        first.reconstruction_loss, second.reconstruction_loss, rtol=0, atol=0
    )
    torch.testing.assert_close(first.consistency_loss, second.consistency_loss, rtol=0, atol=0)


def test_self_supervision_validates_masks_and_optional_view_pairs():
    objective = _objective()
    features = torch.randn(1, 2, 4)
    targets = torch.randn(1, 2, 3)
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="padded"):
        objective(features, mask, targets, torch.tensor([[True, True]]))
    with pytest.raises(ValueError, match="together"):
        objective(features, mask, targets, secondary_features=features)
    with pytest.raises(ValueError, match="match primary"):
        objective(
            features,
            mask,
            targets,
            secondary_features=torch.randn(1, 3, 4),
            secondary_mask=torch.ones(1, 2, dtype=torch.bool),
        )
