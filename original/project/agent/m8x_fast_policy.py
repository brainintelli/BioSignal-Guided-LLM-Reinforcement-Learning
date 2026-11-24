"""Opt-in, unregistered inference/training layouts for bounded GPU diagnostics.

The frozen Policy remains the formal training/serving default. Each compact
input uses a canonical B1 shape, independent of neighboring samples. Prefix
caching shares only the exact tokenized system turn, never the incompatible
verb/target assistant tails. Training caches retain autograd and are local to
one forward graph; inference caches are invalidated by parameter versions.
"""
from contextlib import nullcontext
import copy

import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer

from agent.m8x_contract import RULES
from agent.m8x_fast_layout import split_prefix
from agent.m8x_policy import Policy


def parameter_revision(model):
    """Optimizer and load_state_dict in-place writes advance Tensor._version."""
    return tuple((name, id(p), p.data_ptr(), p._version, p.dtype, p.device)
                 for name, p in model.named_parameters())


class FunctionalLinearLayer(LinearAttentionLayer):
    """Do not overwrite a recurrent tensor saved for a continuation backward."""
    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx,
                                     conv_kernel_size=conv_kernel_size)
        full = (torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
                if self.has_previous_state[state_idx] else conv_states)
        self.has_previous_state[state_idx] = True
        size = self.conv_kernel_size[state_idx]
        if full.shape[-1] < size:
            full = torch.nn.functional.pad(full, (size-full.shape[-1], 0))
        self.conv_states[state_idx] = full[..., -size:]
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()  # Preserve autograd, unlike detach/deepcopy.
    if isinstance(value, list):
        return [_clone_tree(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(x) for x in value)
    if isinstance(value, dict):
        return {k:_clone_tree(v) for k,v in value.items()}
    return value


def fork_cache(cache):
    result = copy.copy(cache)
    result.layers = []
    for layer in cache.layers:
        branch = copy.copy(layer)
        for key,value in vars(layer).items():
            setattr(branch, key, _clone_tree(value))
        result.layers.append(branch)
    return result


class FastPolicy(Policy):
    MODES = ('exact', 'compact', 'prefix')

    def __init__(self, *args, optimization='prefix', **kwargs):
        if optimization not in self.MODES:
            raise ValueError('Unknown optimization layout')
        super().__init__(*args, **kwargs)
        if self.family != 'qwen3_5':
            raise ValueError('This fast path is verified only for the Qwen3.5 backbone')
        self.optimization = optimization
        text = self.tokenizer.apply_chat_template([{'role':'system','content':RULES},
                                                  {'role':'user','content':'prefix boundary probe'}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
        marker = '<|im_start|>user\n'
        if text.count(marker) != 1:
            raise ValueError('Unexpected Qwen user-turn boundary')
        self.prefix_ids = self.tokenizer.encode(text.split(marker)[0], add_special_tokens=False)
        self._inference_prefixes = {}

    def _backbone(self, model, ids, cache=None):
        start = cache.get_seq_length() if cache is not None else 0
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device_name)
        positions = torch.arange(start,start+len(ids),device=self.device_name).view(1,1,-1).expand(3,1,-1)
        return model.get_base_model().model(
            input_ids=inputs, position_ids=positions,
            attention_mask=torch.ones((1,start+len(ids)),device=self.device_name,dtype=torch.long),
            past_key_values=cache, use_cache=cache is not None,
            output_hidden_states=False, return_dict=True)

    def _prefix(self, model, reference):
        inference = not torch.is_grad_enabled()
        revision = parameter_revision(model)
        existing = self._inference_prefixes.get(reference)
        if inference and existing is not None and existing[0] == revision:
            return existing[1]
        cache = DynamicCache(config=model.get_base_model().config)
        cache.layers = [FunctionalLinearLayer() if isinstance(x,LinearAttentionLayer) else x
                        for x in cache.layers]
        self._backbone(model, self.prefix_ids, cache)
        if inference:
            self._inference_prefixes[reference] = (revision,cache)
        return cache

    def _logits(self, inputs, token_ids, reference=False):
        if not inputs or any(not ids or len(ids)>self.max_tokens for ids in inputs):
            raise ValueError('Empty input or context overflow; never truncate legal choices')
        model = self.reference_model if reference else self.model
        if model is None:
            raise RuntimeError('Reference requested from an inference-only policy')
        with torch.no_grad() if reference else nullcontext():
            weights = model.get_output_embeddings().weight[token_ids].float()
            if self.optimization == 'exact':
                chunks = []
                for start in range(0,len(inputs),4):
                    real=inputs[start:start+4]
                    physical=real+[real[-1]]*(4-len(real))
                    batch=self.tokenizer.pad({'input_ids':physical},padding='max_length',
                                             max_length=self.max_tokens,return_tensors='pt')
                    batch={k:v.to(self.device_name) for k,v in batch.items()}
                    out=model.get_base_model().model(**batch,use_cache=False,
                                                    output_hidden_states=False,return_dict=True)
                    chunks.append(torch.nn.functional.linear(out.last_hidden_state[:,-1,:].float(),weights)[:len(real)])
                return torch.cat(chunks,dim=0)
            if self.optimization == 'prefix':
                suffixes=split_prefix(inputs,self.prefix_ids,self.max_tokens)
                shared=self._prefix(model,reference)
            else:
                suffixes=inputs
                shared=None
            chunks=[]
            for ids in suffixes:
                out=self._backbone(model,ids,fork_cache(shared) if shared is not None else None)
                chunks.append(torch.nn.functional.linear(out.last_hidden_state[:,-1,:].float(),weights))
            return torch.cat(chunks,dim=0)

    def metadata(self):
        data=super().metadata()
        data.update(inference_layout='diagnostic_'+self.optimization,
                    physical_forward_shape=([4,1536] if self.optimization=='exact' else 'B1_exact_input_length'),
                    optimization_registered=False,release_authorized=False,
                    prefix_tokens=len(self.prefix_ids),prefix_cache_autograd='graph_local')
        return data

    def save(self, path):
        raise RuntimeError('A new registration and checkpoint identity are required before saving this layout')
