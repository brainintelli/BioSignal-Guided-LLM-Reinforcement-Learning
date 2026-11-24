"""Canonical B4 state buckets, grouped projections, graph-local target KV.

Each state's width is independent of its peers. Padding occurs before the
entire state, never between a live recurrent prefix and its continuation.
The 1536-token limit remains a rejection limit, not a truncation policy.
"""
from contextlib import nullcontext

import torch

from agent.m8s2_cache import fresh_cache
from agent.m8s3_batch_layout import canonical_batches
from agent.m8s3_padded_policy import PaddedSemanticPolicy, select_cache_row


class CanonicalBatchPolicy(PaddedSemanticPolicy):
    bucket_boundaries = (896, 1280, 1537)

    def _roots(self, prepared, reference):
        roots = self._kv_scope['roots']
        missing = list(dict.fromkeys(tuple(row['input_ids']) for row in prepared
                                     if roots[tuple(row['input_ids'])] is None))
        model = self.reference_model if reference else self.model
        with torch.no_grad() if reference else nullcontext():
            if missing:
                for batch in canonical_batches(missing, self.bucket_boundaries,
                                               pad_id=self.tokenizer.pad_token_id):
                    ids = torch.tensor(batch['input_ids'], device=self.device_name)
                    mask = torch.tensor(batch['attention_mask'], device=self.device_name)
                    positions = (mask.cumsum(-1) - 1).clamp_min(0).unsqueeze(0).expand(3, -1, -1)
                    cache = fresh_cache(model.get_base_model().config)
                    out = model.get_base_model().model(
                        input_ids=ids, attention_mask=mask, position_ids=positions,
                        past_key_values=cache, use_cache=True, output_hidden_states=False, return_dict=True)
                    for physical, original in enumerate(batch['indices']):
                        key = missing[original]
                        child = select_cache_row(cache, physical)
                        child.state_padding = batch['width'] - len(key)
                        roots[key] = child, out.last_hidden_state[physical:physical + 1, -1, :]
                    self.cache_stats['forward_calls'] += 1
                    self.cache_stats['forward_tokens'] += 4 * batch['width']
        return [roots[tuple(row['input_ids'])] for row in prepared]

    def metadata(self):
        result = super().metadata()
        result.update(inference_layout='semantic_v3_canonical_state_buckets_b4',
                      physical_forward_shape={'batch': 4, 'widths': list(self.bucket_boundaries)},
                      bucket_rule='smallest registered width > individual state length; always a masked leading token',
                      default_prefix_layout='full_state_bucket',
                      default_layout_reason='Batch whole states to reduce backbone dispatch overhead',
                      comparison_scope='native B4x1536 vs grouped canonical B4 buckets; identical target KV algorithm',
                      system_prefix_cache_active=False)
        return result
