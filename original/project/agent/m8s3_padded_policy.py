"""Fixed B4x1536 state prefills with efficient graph-local semantic target KV.

Both arms share fused projections and target branching. This conservative
control isolates state padding/static-prefix reuse, without recomputing an
entire padded state for every semantic trie node.
"""
import copy
from contextlib import contextmanager,nullcontext
import torch
from agent.m8s3_policy import BatchedSemanticPolicy
from agent.m8s3_diag_contract import physical_batches
from agent.m8s2_cache import fresh_cache

def select_cache_row(cache,index,batch=4):
    def select(value):
        if isinstance(value,torch.Tensor):
            if value.ndim and value.shape[0]==batch:return value[index:index+1]
            return value
        if isinstance(value,list):return [select(x) for x in value]
        if isinstance(value,tuple):return tuple(select(x) for x in value)
        if isinstance(value,dict):return {k:select(v) for k,v in value.items()}
        return value
    result=copy.copy(cache);result.layers=[]
    for layer in cache.layers:
        child=copy.copy(layer)
        for key,value in vars(layer).items():setattr(child,key,select(value))
        if hasattr(child,'batch_size'):child.batch_size=1
        result.layers.append(child)
    return result

class PaddedSemanticPolicy(BatchedSemanticPolicy):
    @contextmanager
    def _scope(self,prepared,reference):
        previous=self._kv_scope
        if previous is None or previous['reference']!=reference:self._prepare_execution(reference)
        keys=[tuple(r['input_ids']) for r in prepared]
        if previous is not None and previous['reference']==reference and all(k in previous['roots'] for k in keys):
            yield;return
        self._kv_scope=dict(reference=reference,roots={k:None for k in keys})
        try:yield
        finally:self._kv_scope=previous

    def _roots(self,prepared,reference):
        roots=self._kv_scope['roots'];missing=[]
        for row in prepared:
            key=tuple(row['input_ids'])
            if roots[key] is None and key not in missing:missing.append(key)
        model=self.reference_model if reference else self.model;offset=0
        with torch.no_grad() if reference else nullcontext():
            if missing:
                for batch in physical_batches([list(k) for k in missing],self.max_tokens,self.tokenizer.pad_token_id):
                    ids=torch.tensor(batch['input_ids'],device=self.device_name);mask=torch.tensor(batch['attention_mask'],device=self.device_name)
                    positions=(mask.cumsum(-1)-1).clamp_min(0).unsqueeze(0).expand(3,-1,-1)
                    cache=fresh_cache(model.get_base_model().config)
                    out=model.get_base_model().model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=cache,
                        use_cache=True,output_hidden_states=False,return_dict=True)
                    for i in range(batch['real_rows']):
                        key=missing[offset+i];child=select_cache_row(cache,i);child.state_padding=self.max_tokens-len(key)
                        roots[key]=child,out.last_hidden_state[i:i+1,-1,:]
                    offset+=batch['real_rows'];self.cache_stats['forward_calls']+=1;self.cache_stats['forward_tokens']+=4*self.max_tokens
        return [roots[tuple(row['input_ids'])] for row in prepared]

    def _root(self,ids,reference):
        key=tuple(ids);roots=self._kv_scope['roots']
        if roots[key] is None:self._roots([dict(input_ids=ids)],reference)
        return roots[key]

    def _forward_batch(self,model,sequences,cache):
        padding=cache.state_padding;physical_start=cache.get_seq_length();logical_start=physical_start-padding
        lengths=[len(s) for s in sequences];width=max(2,max(lengths));batch=len(lengths)
        if min(lengths)<1 or logical_start+width>self.max_tokens:raise ValueError('Invalid continuation; never truncate')
        ids=torch.tensor([s+[self.tokenizer.pad_token_id]*(width-len(s)) for s in sequences],device=self.device_name)
        mask=torch.tensor([[0]*padding+[1]*(logical_start+n)+[0]*(width-n) for n in lengths],device=self.device_name)
        positions=torch.arange(logical_start,logical_start+width,device=self.device_name).view(1,1,-1).expand(3,batch,-1)
        out=model.get_base_model().model(input_ids=ids,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
        self.cache_stats['forward_calls']+=1;self.cache_stats['forward_tokens']+=batch*width
        return out.last_hidden_state[torch.arange(batch,device=self.device_name),torch.tensor(lengths,device=self.device_name)-1,:]

    def verb_distribution(self,prepared,reference=False):
        with self._scope(prepared,reference), (torch.no_grad() if reference else nullcontext()):
            model=self.reference_model if reference else self.model
            weights=model.get_output_embeddings().weight[self.verb_ids].float()
            # Canonical B1 head GEMM, matching the unpadded policy. A B4 FP32
            # head otherwise changes rounding when audit rows are regrouped.
            logits=torch.cat([torch.nn.functional.linear(h.float(),weights) for cache,h in self._roots(prepared,reference)])
            mask=torch.tensor([r['verb_mask'] for r in prepared],device=self.device_name,dtype=torch.bool)
            return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-1e9))

    def metadata(self):
        result=super().metadata()
        result.update(inference_layout='semantic_v3_grouped_fixed_state_batch4_1536',physical_forward_shape=[4,1536],
            cache_lifetime='graph-local state/target KV only; no persistent/static prefix cache',scene_cache_active=False,
            native_lora=False,root_hidden_shared=True,target_prefix_layout='same canonical batched short suffixes in both arms',
            comparison_scope='same grouped/fused projections and target branching; fixed state padding vs unpadded/static KV')
        return result

class NativePaddedSemanticPolicy(PaddedSemanticPolicy):
    """Original PEFT arithmetic with the same fixed state and target KV layout."""
    def _enable_execution(self):
        from agent.m8x_grouped_lora import ExecutionCache
        self.execution_cache=ExecutionCache();self.execution_parts={}
        self.optimized_projection_counts={};self.optimized_norm_counts={};self.group_counts={};self.fused_mlp_counts={}

    def _prepare_execution(self,reference):
        pass

    def metadata(self):
        result=super().metadata()
        result.update(inference_layout='semantic_v3_native_fixed_state_batch4_1536',native_lora=True,
                      comparison_scope='native fixed state vs grouped unpadded/static KV; same graph-local target branching')
        return result
