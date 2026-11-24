"""Two-level KV reuse: shared system prefix, then observation and action tail."""
from contextlib import nullcontext
import torch
from transformers.cache_utils import DynamicCache,LinearAttentionLayer
from agent.m8x_fast_policy import FunctionalLinearLayer,parameter_revision
from agent.m8x_kv_policy import branch_cache
from agent.m8x_kv_layout import checked_suffix
from agent.m8x_optimized_policy import OptimizedPolicy


class PrefixBank:
    def __init__(self):self.inference={}

    def acquire(self,reference,revision,scope,build):
        if torch.is_grad_enabled():
            roots=scope.setdefault('system_roots',{})
            if reference not in roots:roots[reference]=build()
            return roots[reference]
        previous=self.inference.get(reference)
        if previous is None or previous[0]!=revision:
            previous=(revision,build())
            self.inference[reference]=previous
        return previous[1]


class SharedSystemPolicy(OptimizedPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.system_bank=PrefixBank()

    def _logits(self,inputs,token_ids,reference=False):
        scope=self._kv_scope
        if scope is None or scope['reference']!=reference:
            raise RuntimeError('Missing graph-local observation scope')
        model=self.reference_model if reference else self.model
        if model is None:raise RuntimeError('No reference model')
        roots=scope['roots'];chunks=[]
        with torch.no_grad() if reference else nullcontext():
            def build():
                cache=DynamicCache(config=model.get_base_model().config)
                cache.layers=[FunctionalLinearLayer() if isinstance(x,LinearAttentionLayer) else x for x in cache.layers]
                self._backbone(model,self.prefix_ids,cache)
                return cache
            # Revisions matter only for caches that survive this graph.
            revision=parameter_revision(model) if not torch.is_grad_enabled() else None
            system=self.system_bank.acquire(reference,revision,scope,build)
            weights=model.get_output_embeddings().weight[token_ids].float()
            for ids in inputs:
                matches=[p for p in roots if ids[:len(p)]==list(p)]
                if len(matches)!=1:raise ValueError('Ambiguous observation prefix')
                prefix=matches[0]
                if roots[prefix] is None:
                    suffix=checked_suffix(list(prefix),self.prefix_ids,self.max_tokens)
                    cache=branch_cache(system)
                    out=self._backbone(model,suffix,cache)
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
        data.update(inference_layout='unpadded_two_level_kv_merged_v2',
            system_prefix_tokens=len(self.prefix_ids),
            system_cache_lifetime='weight_version_inference_or_one_gradient_graph',
            cache_tensor_copy=False)
        return data
