"""Target-free multi-paradigm task heads and training bookkeeping contracts."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class TaskTrackConfig:
    name:str; modalities:tuple[str,...]; num_classes:int
    def __post_init__(self):
        if not self.name.strip() or not self.modalities or any(not x.strip() for x in self.modalities) or self.num_classes<2: raise ValueError("task track requires name, modalities, and at least two classes")
@dataclass(frozen=True)
class MultiParadigmTaskConfig:
    hidden_size:int; tracks:tuple[TaskTrackConfig,...]
    def __post_init__(self):
        if self.hidden_size<1 or not self.tracks or len({x.name for x in self.tracks})!=len(self.tracks): raise ValueError("task config requires positive hidden size and unique tracks")
@dataclass(frozen=True)
class MultiTaskTrainingOutput:
    logits:torch.Tensor; loss:torch.Tensor; task_name:str; modality:str

class MultiParadigmTaskHead(nn.Module):
    """Task-token classifier; forward contains no labels and is inference-safe."""
    def __init__(self, config:MultiParadigmTaskConfig):
        super().__init__(); self.config=config; self.names=tuple(x.name for x in config.tracks); self.tracks={x.name:x for x in config.tracks}
        self.task_tokens=nn.Parameter(torch.zeros(len(self.names),config.hidden_size)); nn.init.normal_(self.task_tokens,std=config.hidden_size**-0.5)
        self.classifiers=nn.ModuleDict({x.name:nn.Linear(config.hidden_size,x.num_classes) for x in config.tracks})
    def _pooled(self,features,mask):
        if features.ndim!=3 or features.shape[-1]!=self.config.hidden_size: raise ValueError("features must be [batch, tokens, hidden_size]")
        if mask.shape!=features.shape[:2] or not mask.bool().any(dim=1).all(): raise ValueError("mask must match features and retain valid tokens")
        w=mask.to(features.dtype); return (features*w.unsqueeze(-1)).sum(1)/w.sum(1,keepdim=True)
    def _track(self,task_name,modality):
        if task_name not in self.tracks: raise ValueError("unknown task_name")
        track=self.tracks[task_name]
        if modality not in track.modalities: raise ValueError("modality is incompatible with task_name")
        return track,self.names.index(task_name)
    def forward(self,features,mask,task_name:str,modality:str):
        _,idx=self._track(task_name,modality); return self.classifiers[task_name](self._pooled(features,mask)+self.task_tokens[idx])
    def training_loss(self,features,mask,task_name,modality,labels):
        track,_=self._track(task_name,modality); logits=self(features,mask,task_name,modality)
        if labels.ndim!=1 or labels.shape[0]!=features.shape[0] or labels.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8): raise ValueError("labels must be integer [batch]")
        labels=labels.to(logits.device,dtype=torch.long)
        if labels.lt(0).any() or labels.ge(track.num_classes).any(): raise ValueError("labels must belong to task class vocabulary")
        return MultiTaskTrainingOutput(logits,F.cross_entropy(logits,labels),task_name,modality)

@dataclass(frozen=True)
class BalancedTaskSchedule:
    task_names:tuple[str,...]
    def __post_init__(self):
        if not self.task_names or len(set(self.task_names))!=len(self.task_names): raise ValueError("balanced schedule requires unique nonempty task names")
    def task_at(self,step:int)->str:
        if not isinstance(step,int) or step<0: raise ValueError("step must be nonnegative integer")
        return self.task_names[step%len(self.task_names)]

@dataclass(frozen=True)
class GradientConflictLog:
    pairwise_cosines:dict[tuple[str,str],float]; mean_cosine:float; negative_pairs:int
def gradient_conflict_log(task_gradients:Mapping[str,torch.Tensor])->GradientConflictLog:
    if not task_gradients: raise ValueError("task_gradients cannot be empty")
    names=sorted(task_gradients); grads=[]
    for name in names:
        value=task_gradients[name]
        if value.ndim!=1 or not torch.is_floating_point(value) or not torch.isfinite(value).all(): raise ValueError("each task gradient must be finite one-dimensional floating tensor")
        grads.append(value)
    if len({x.numel() for x in grads})!=1: raise ValueError("task gradients must have equal length")
    pairs={}
    for (left,lg),(right,rg) in combinations(zip(names,grads),2):
        pairs[(left,right)]=float(F.cosine_similarity(lg,rg,dim=0,eps=1e-12).item())
    return GradientConflictLog(pairs,sum(pairs.values())/len(pairs) if pairs else 1.0,sum(x<0 for x in pairs.values()))
