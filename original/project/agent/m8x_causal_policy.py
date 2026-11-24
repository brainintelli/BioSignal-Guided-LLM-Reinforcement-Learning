"""Full-verb prefill reused by an explicitly normalized target continuation."""
from contextlib import nullcontext
import torch
from transformers.cache_utils import DynamicCache,LinearAttentionLayer
from agent.m8x_fast_policy import FastPolicy,FunctionalLinearLayer
from agent.m8x_kv_policy import KVPolicy,branch_cache
from agent.m8x_causal_kv import normalize_target
from agent.m8x_kv_layout import checked_suffix


class CausalTemplate:
    def prepare(self,observations,salt='m6'):
        prepared=FastPolicy.prepare(self,observations,salt)
        stub=self.tokenizer.encode('<think>\n\n</think>\n\n',add_special_tokens=False)
        for row in prepared:
            if row['input_ids'][-len(stub):]!=stub:raise ValueError('Unexpected disabled-thinking stub')
            row['kv_prefix_ids']=row['input_ids']
            row['assistant_boundary']=len(row['input_ids'])-len(stub)
            row['normalized_targets']={}
        return prepared

    def _target_tokens(self,row,verb):
        if verb not in row['normalized_targets']:
            original=FastPolicy._target_tokens(self,row,verb)
            row['normalized_targets'][verb]=normalize_target(row['input_ids'],original,row['assistant_boundary'],self.max_tokens)
        return row['normalized_targets'][verb]


class CausalCompactPolicy(CausalTemplate,FastPolicy):
    def __init__(self,*args,**kwargs):super().__init__(*args,optimization='compact',**kwargs)


class CausalKVPolicy(CausalTemplate,KVPolicy):
    def _logits(self,inputs,token_ids,reference=False):
        if self._kv_scope is None or self._kv_scope['reference']!=reference:
            raise RuntimeError('Missing graph-local KV scope')
        model=self.reference_model if reference else self.model
        if model is None:raise RuntimeError('No reference model')
        roots=self._kv_scope['roots'];chunks=[]
        with torch.no_grad() if reference else nullcontext():
            weights=model.get_output_embeddings().weight[token_ids].float()
            for ids in inputs:
                matches=[p for p in roots if ids[:len(p)]==list(p)]
                if len(matches)!=1:raise ValueError('Ambiguous full-verb prefix')
                prefix=matches[0]
                if roots[prefix] is None:
                    cache=DynamicCache(config=model.get_base_model().config)
                    cache.layers=[FunctionalLinearLayer() if isinstance(x,LinearAttentionLayer) else x for x in cache.layers]
                    out=self._backbone(model,list(prefix),cache)
                    roots[prefix]=(cache,out.last_hidden_state[:,-1,:])
                cache,last=roots[prefix]
                if len(ids)!=len(prefix):
                    suffix=checked_suffix(ids,prefix,self.max_tokens)
                    out=self._backbone(model,suffix,branch_cache(cache))
                    last=out.last_hidden_state[:,-1,:]
                chunks.append(torch.nn.functional.linear(last.float(),weights))
        return torch.cat(chunks,dim=0)

    def metadata(self):
        data=super().metadata()
        data.update(inference_layout='unpadded_full_verb_kv_v1',
                    target_template='explicit_empty_thinking_previous_assistant_v1',
                    prefix_cache_autograd='graph_local_shared_full_verb')
        return data
