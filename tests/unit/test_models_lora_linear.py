import torch
from torch import nn

from openthought2text.models import (
    LoRAAdapterConfig,
    LoRALinearAdapter,
    LoRAProvenance,
    LoRAScheduleConfig,
)


def test_lora_adapter_freezes_base_zero_scale_and_masks():
    adapter = LoRALinearAdapter(
        nn.Linear(3, 2),
        LoRAAdapterConfig(2, 4.0, 0.0, ("q",)),
        LoRAProvenance("base", "fp", "unknown"),
        LoRAScheduleConfig(1, 2, 1.0),
    )
    values = torch.randn(1, 3, 3, requires_grad=True)
    base = adapter.base_projection(values).detach()
    torch.testing.assert_close(adapter(values, scale=0), base)
    adapter(values).sum().backward()
    assert (
        all(p.grad is None for p in adapter.base_projection.parameters())
        and adapter.down.weight.grad is not None
        and adapter.up.weight.grad is not None
    )
    changed = values.detach().clone()
    changed[:, 1] = 1000
    mask = torch.tensor([[True, False, True]])
    torch.testing.assert_close(adapter(values.detach(), mask), adapter(changed, mask))
    assert torch.all(adapter(values.detach(), mask)[:, 1] == 0)
