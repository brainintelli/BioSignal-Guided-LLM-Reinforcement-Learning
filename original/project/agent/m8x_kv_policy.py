"""Unpadded, graph-local sharing of exact observation prefixes across stages.

Cache tensors are immutable between branches: attention updates concatenate,
and recurrent/convolution updates assign new tensors. No cache survives a
public distribution/act call, so optimizer steps cannot reuse stale states.
"""
import copy
from contextlib import contextmanager, nullcontext

import torch
from transformers.cache_utils import DynamicCache,DynamicLayer,LinearAttentionLayer

from agent.m8x_fast_policy import FastPolicy,FunctionalLinearLayer
from agent.m8x_kv_layout import checked_suffix


def _containers(value):
    if isinstance(value,list):return [_containers(x) for x in value]
    if isinstance(value,dict):return {k:_containers(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_containers(x) for x in value)
    return value


def branch_cache(cache):
    """Copy cache containers, share read-only tensors and their autograd graph."""
    branch=copy.copy(cache)
    branch.layers=[]
    for layer in cache.layers:
        if type(layer) not in (DynamicLayer,FunctionalLinearLayer):
            raise TypeError('Only verified non-mutating cache layers may be shared')
        child=copy.copy(layer)
        for name,value in vars(layer).items():setattr(child,name,_containers(value))
        branch.layers.append(child)
    return branch


class KVPolicy(FastPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,optimization='compact',**kwargs)
        self._kv_scope=None

    def prepare(self,observations,salt='m6'):
        prepared=super().prepare(observations,salt)
        for row in prepared:
            text=self.tokenizer.apply_chat_template(row['messages'],tokenize=False,
                    add_generation_prompt=False,enable_thinking=False)
            prefix=self.tokenizer.encode(text,add_special_tokens=False)
            checked_suffix(row['input_ids'],prefix,self.max_tokens)
            row['kv_prefix_ids']=prefix
        return prepared

    @contextmanager
    def _scope(self,prepared,reference):
        previous=self._kv_scope
        prefixes=tuple(tuple(row['kv_prefix_ids']) for row in prepared)
        if previous is not None and previous['reference']==reference and all(p in previous['roots'] for p in prefixes):
            yield
            return
        self._kv_scope=dict(reference=reference,roots={p:None for p in prefixes})
        try:yield
        finally:self._kv_scope=previous

    def _logits(self,inputs,token_ids,reference=False):
        if self._kv_scope is None or self._kv_scope['reference']!=reference:
            raise RuntimeError('KV forwards require a graph-local observation scope')
        model=self.reference_model if reference else self.model
        if model is None:raise RuntimeError('No reference model')
        roots=self._kv_scope['roots']
        chunks=[]
        with torch.no_grad() if reference else nullcontext():
            weights=model.get_output_embeddings().weight[token_ids].float()
            for ids in inputs:
                matches=[p for p in roots if ids[:len(p)]==list(p)]
                if len(matches)!=1:raise ValueError('Input must have exactly one registered state prefix')
                prefix=matches[0]
                suffix=checked_suffix(ids,prefix,self.max_tokens)
                cache=roots[prefix]
                if cache is None:
                    cache=DynamicCache(config=model.get_base_model().config)
                    cache.layers=[FunctionalLinearLayer() if isinstance(x,LinearAttentionLayer) else x for x in cache.layers]
                    self._backbone(model,list(prefix),cache)
                    roots[prefix]=cache
                out=self._backbone(model,suffix,branch_cache(cache))
                chunks.append(torch.nn.functional.linear(out.last_hidden_state[:,-1,:].float(),weights))
        return torch.cat(chunks,dim=0)

    def verb_distribution(self,prepared,reference=False):
        with self._scope(prepared,reference):return super().verb_distribution(prepared,reference)

    def target_distribution(self,prepared,verbs,reference=False):
        with self._scope(prepared,reference):return super().target_distribution(prepared,verbs,reference)

    def distributions(self,prepared,verbs,reference=False):
        with self._scope(prepared,reference):return super().distributions(prepared,verbs,reference)

    @torch.no_grad()
    def act(self,prepared,greedy=False,with_reference=False):
        with self._scope(prepared,False):return super().act(prepared,greedy,with_reference)

    def metadata(self):
        data=super().metadata()
        data.update(inference_layout='unpadded_observation_kv_v1',
                    physical_forward_shape='B1_exact_prefix_and_suffix',
                    prefix_cache_autograd='graph_local_shared_observation',
                    cache_tensor_copy=False,cache_lifetime='one_distribution_or_act_call')
        return data
