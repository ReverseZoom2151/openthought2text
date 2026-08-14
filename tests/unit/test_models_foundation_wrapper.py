import pytest
import torch
from torch import nn

from openthought2text.models import (
    FoundationEncoderWrapper,
    FoundationFeatureContract,
    FoundationPretrainingProvenance,
    TokenTiming,
)


class _ExternalFeatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 6)

    def forward(self, features, mask):
        return self.projection(features)


def _timing(batch=1, tokens=3):
    start = torch.arange(tokens).unsqueeze(0).expand(batch, -1)
    return TokenTiming(start, start + 1, 100.0)


def _wrapper(trainable=False):
    return FoundationEncoderWrapper(
        _ExternalFeatureModule(),
        FoundationFeatureContract(4, 6),
        FoundationPretrainingProvenance("external-demo", "unknown", "declared external neural pretraining"),
        trainable=trainable,
    )


def test_foundation_wrapper_passes_through_masks_timing_and_makes_padded_values_inert():
    torch.manual_seed(32)
    wrapper = _wrapper().eval()
    mask = torch.tensor([[True, True, False]])
    features = torch.randn(1, 3, 4)
    changed = features.clone()
    changed[:, 2] = 1_000
    timing = _timing()
    first = wrapper(features, mask, timing)
    second = wrapper(changed, mask, timing)
    assert first.features.shape == (1, 3, 6)
    assert first.mask.equal(mask) and first.timing is timing
    assert torch.all(first.features[:, 2] == 0)
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)


def test_foundation_wrapper_freezes_or_enables_external_parameter_gradients():
    frozen = _wrapper(trainable=False)
    assert not frozen.trainable and not any(item.requires_grad for item in frozen.external_encoder.parameters())
    features = torch.randn(1, 2, 4, requires_grad=True)
    frozen(features, torch.ones(1, 2, dtype=torch.bool), _timing(tokens=2)).features.sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert all(item.grad is None for item in frozen.external_encoder.parameters())
    frozen.set_trainable(True)
    assert frozen.trainable and all(item.requires_grad for item in frozen.external_encoder.parameters())
    frozen.zero_grad(set_to_none=True)
    frozen(torch.randn(1, 2, 4), torch.ones(1, 2, dtype=torch.bool), _timing(tokens=2)).features.sum().backward()
    assert any(item.grad is not None for item in frozen.external_encoder.parameters())


def test_foundation_wrapper_validates_contract_provenance_and_external_output():
    with pytest.raises(ValueError, match="overlap_label"):
        FoundationPretrainingProvenance("source", "undeclared", "description")
    with pytest.raises(ValueError, match="feature_layout"):
        FoundationFeatureContract(4, 6, feature_layout="channels_time")
    wrapper = _wrapper()
    with pytest.raises(ValueError, match="input_feature_size"):
        wrapper(torch.randn(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), _timing(tokens=2))
    invalid = FoundationEncoderWrapper(
        nn.Identity(), FoundationFeatureContract(4, 6), FoundationPretrainingProvenance("x", "disjoint", "y")
    )
    with pytest.raises(ValueError, match="forward\\(features, mask\\)"):
        invalid(torch.randn(1, 2, 4), torch.ones(1, 2, dtype=torch.bool), _timing(tokens=2))
