"""Replicate hybrid caches across semantic branches without shared mutation."""
import copy
import torch
from transformers.cache_utils import DynamicCache,LinearAttentionLayer
from agent.m8x_fast_policy import FunctionalLinearLayer

def fresh_cache(config):
    cache=DynamicCache(config=config)
    cache.layers=[FunctionalLinearLayer() if isinstance(x,LinearAttentionLayer) else x for x in cache.layers]
    for layer in cache.layers:
        if isinstance(layer,FunctionalLinearLayer):layer.record_past=True
    return cache

def repeat_cache(cache,batch):
    if batch<1:raise ValueError('A branch batch cannot be empty')
    def replicate(value):
        if isinstance(value,torch.Tensor):
            if value.ndim==0:return value.clone()
            if value.shape[0]!=1:raise ValueError('Only single-state roots can be replicated')
            return value.repeat((batch,)+(1,)*(value.ndim-1))
        if isinstance(value,list):return [replicate(v) for v in value]
        if isinstance(value,dict):return {k:replicate(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(replicate(v) for v in value)
        return value
    result=copy.copy(cache);result.layers=[]
    for layer in cache.layers:
        child=copy.copy(layer)
        for key,value in vars(layer).items():setattr(child,key,replicate(value))
        if hasattr(child,'batch_size'):child.batch_size=batch
        if isinstance(child,FunctionalLinearLayer):child.record_past=True
        result.layers.append(child)
    return result
