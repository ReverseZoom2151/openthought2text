import inspect
import pytest
import torch
from openthought2text.models import (LoRAAdapterConfig, LoRAAdapterContract, LoRAProvenance, LoRAScheduleConfig, NeuralFeatureProjector, NeuralProjectorConfig, OrderedSemanticAnchorDecoder, OrderedSemanticAnchorDecoderConfig, OrderedSemanticAnchorGenerationConfig)

def _decoder(): return OrderedSemanticAnchorDecoder(OrderedSemanticAnchorDecoderConfig(num_anchors=5, hidden_size=8, num_layers=1, num_heads=2, max_sequence_length=6, dropout=0)).eval()
def test_anchor_decoder_training_masks_targets_grad_and_generation_is_target_free():
    decoder=_decoder(); features=torch.randn(2,3,8,requires_grad=True); mask=torch.tensor([[True,True,False],[True,True,True]])
    trained=decoder.training_forward(features,mask,torch.tensor([[1,2,0],[3,4,1]]),torch.tensor([[True,True,False],[True,True,True]]))
    assert trained.logits.shape==(2,3,5); trained.loss.backward(); assert features.grad is not None and features.grad.abs().sum()>0
    assert set(inspect.signature(decoder.generate).parameters)=={"neural_features","neural_mask","config"}
    generated=decoder.generate(features.detach(),mask,OrderedSemanticAnchorGenerationConfig(max_new_anchors=3)); assert generated.anchor_ids.shape[0]==2 and generated.anchor_mask.shape==generated.anchor_ids.shape
def test_projector_masks_padding_and_lora_contract_is_validated():
    projector=NeuralFeatureProjector(NeuralProjectorConfig(3,4,"mlp",0)).eval(); values=torch.randn(1,3,3,requires_grad=True); mask=torch.tensor([[True,False,True]])
    output=projector(values,mask); assert output.shape==(1,3,4) and torch.all(output[:,1]==0); output.sum().backward(); assert values.grad is not None and torch.all(values.grad[:,1]==0)
    contract=LoRAAdapterContract(LoRAAdapterConfig(2,4.0,0.0,("decoder.q",)),LoRAScheduleConfig(2,2,1.0),LoRAProvenance("base","fingerprint","unknown"))
    assert [contract.training_scale(x) for x in range(5)]==[0.0,0.0,0.5,1.0,1.0]
def test_anchor_and_lora_validation():
    with pytest.raises(ValueError,match="divide"): OrderedSemanticAnchorDecoderConfig(5,7,num_heads=2)
    with pytest.raises(ValueError,match="explicit"): LoRAProvenance("base","fp","bad")
    with pytest.raises(ValueError,match="target_module"): LoRAAdapterConfig(1,1.0,0.0,())
    with pytest.raises(ValueError,match="vocabulary"):
        _decoder().training_forward(torch.randn(1,2,8),torch.ones(1,2,dtype=torch.bool),torch.tensor([[8]]))
