import inspect

import pytest
import torch
from torch.nn import functional as F

from openthought2text.models import CrossSubjectDomainAdversary


def test_domain_adversary_masks_padding_exposes_train_only_labels_and_backpropagates():
    torch.manual_seed(25)
    adversary = CrossSubjectDomainAdversary(hidden_size=4, num_subjects=3)
    features = torch.randn(2, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    output = adversary.training_loss(
        features, mask, torch.tensor([0, 2]), gradient_reversal_scale=0.5
    )
    assert output.logits.shape == (2, 3)
    assert output.pooled_features.shape == (2, 4)
    assert output.valid_token_counts.tolist() == [2, 2]
    assert set(inspect.signature(adversary.forward).parameters) == {
        "neural_features",
        "token_mask",
        "gradient_reversal_scale",
    }
    output.loss.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_gradient_reversal_negates_feature_gradient_but_keeps_classifier_path():
    torch.manual_seed(26)
    adversary = CrossSubjectDomainAdversary(hidden_size=3, num_subjects=2)
    labels = torch.tensor([1])
    mask = torch.ones(1, 2, dtype=torch.bool)
    adversarial_features = torch.randn(1, 2, 3, requires_grad=True)
    adversarial_loss = adversary.training_loss(adversarial_features, mask, labels).loss
    adversarial_loss.backward()
    reversed_gradient = adversarial_features.grad.clone()
    plain_features = adversarial_features.detach().clone().requires_grad_()
    pooled = plain_features.mean(dim=1)
    plain_loss = F.cross_entropy(adversary.classifier(pooled), labels)
    plain_loss.backward()
    torch.testing.assert_close(reversed_gradient, -plain_features.grad)


def test_domain_adversary_rejects_missing_invalid_subjects_and_empty_masks():
    adversary = CrossSubjectDomainAdversary(hidden_size=2, num_subjects=2)
    features = torch.randn(1, 2, 2)
    with pytest.raises(ValueError, match="subject_ids"):
        adversary.training_loss(features, torch.ones(1, 2, dtype=torch.bool), torch.tensor([2]))
    with pytest.raises(ValueError, match="integer"):
        adversary.training_loss(features, torch.ones(1, 2, dtype=torch.bool), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="at least one"):
        adversary(features, torch.zeros(1, 2, dtype=torch.bool))
