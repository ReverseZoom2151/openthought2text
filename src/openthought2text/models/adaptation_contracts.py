"""Target-free projector and LoRA schedule/provenance contracts; no loaders."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

@dataclass(frozen=True)
class NeuralProjectorConfig:
    input_size:int; output_size:int; kind:Literal["linear","mlp"]="linear"; dropout:float=0.0
    def __post_init__(self):
        if self.input_size < 1 or self.output_size < 1 or self.kind not in {"linear","mlp"} or not 0 <= self.dropout < 1: raise ValueError("invalid projector configuration")
class NeuralFeatureProjector(nn.Module):
    def __init__(self, config:NeuralProjectorConfig):
        super().__init__(); self.config=config
        self.network=nn.Linear(config.input_size,config.output_size) if config.kind=="linear" else nn.Sequential(nn.Linear(config.input_size,config.input_size),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(config.input_size,config.output_size))
    def forward(self,features,mask):
        if features.ndim!=3 or features.shape[-1]!=self.config.input_size: raise ValueError("features must be [batch, tokens, projector input_size]")
        if mask.shape!=features.shape[:2]: raise ValueError("mask must be [batch, tokens]")
        valid=mask.bool(); return self.network(features*valid.unsqueeze(-1).to(features.dtype))*valid.unsqueeze(-1).to(features.dtype)

@dataclass(frozen=True)
class LoRAAdapterConfig:
    rank:int; alpha:float; dropout:float; target_module_names:tuple[str,...]
    def __post_init__(self):
        if self.rank<1 or self.alpha<=0 or not 0<=self.dropout<1: raise ValueError("LoRA rank/alpha/dropout are invalid")
        if not self.target_module_names or any(not isinstance(x,str) or not x.strip() for x in self.target_module_names): raise ValueError("target_module_names must contain nonempty declared module names")
@dataclass(frozen=True)
class LoRAScheduleConfig:
    warmup_steps:int=0; ramp_steps:int=0; max_scale:float=1.0
    def __post_init__(self):
        if self.warmup_steps<0 or self.ramp_steps<0 or self.max_scale<0: raise ValueError("LoRA schedule values must be nonnegative")
    def scale_at(self,global_step:int)->float:
        if not isinstance(global_step,int) or global_step<0: raise ValueError("global_step must be a nonnegative integer")
        if global_step<self.warmup_steps:return 0.0
        return self.max_scale if self.ramp_steps==0 else self.max_scale*min(1.0,(global_step-self.warmup_steps+1)/self.ramp_steps)
@dataclass(frozen=True)
class LoRAProvenance:
    base_model_identifier:str; base_model_fingerprint:str; pretraining_overlap_label:str
    def __post_init__(self):
        if not self.base_model_identifier.strip() or not self.base_model_fingerprint.strip(): raise ValueError("LoRA provenance requires nonempty base model identifier and fingerprint")
        if self.pretraining_overlap_label not in {"disjoint","unknown","potential_overlap","same_dataset"}: raise ValueError("pretraining_overlap_label must be explicit and valid")
@dataclass(frozen=True)
class LoRAAdapterContract:
    config:LoRAAdapterConfig; schedule:LoRAScheduleConfig; provenance:LoRAProvenance
    def training_scale(self,global_step:int)->float:return self.schedule.scale_at(global_step)
