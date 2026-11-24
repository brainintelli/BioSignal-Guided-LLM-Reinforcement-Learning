"""Catch fact loss, cross-branch cache mutation and detached cache gradients."""
import importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class CompactSemanticTests(unittest.TestCase):
    def test_real_observations_keep_actions_and_facts_with_less_repetition(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8s2_contract'),'Compact semantic contract missing')
        from agent.m8s2_contract import encode
        from agent.m8s_contract import encode as previous
        dataset=ROOT/'output/rl_semantic_20261108/dataset.json'
        if not dataset.exists():self.skipTest('Real evidence dataset absent')
        rows=json.loads(dataset.read_text(encoding='utf-8'))['observations'][:384]
        before=after=0
        for item in rows:
            old=previous(item['observation']);new=encode(item['observation'])
            for key in ('branches','verb_mask','features','target_texts','semantic_state'):
                self.assertEqual(new[key],old[key],key)
            for targets in new['target_texts']:
                for label in targets:
                    if label:self.assertIn(label,new['messages'][1]['content'])
            before+=sum(len(m['content']) for m in old['messages'])
            after+=sum(len(m['content']) for m in new['messages'])
        self.assertLess(after,before*.85,'Repeated prose still dominates input')

    def test_map_groups_keep_each_exception_and_its_kind(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8s2_contract'),'Compact semantic contract missing')
        from agent.m8s2_contract import terrain_text
        text=terrain_text({'未列位置均为':'草地','例外位置':{'右2格':'水面不可走','右2格、上1格':'水面不可走','左1格':'实体阻挡'}})
        self.assertIn('水面不可走：右2格；右2格、上1格',text)
        self.assertIn('实体阻挡：左1格',text)
        self.assertIn('其余草地',text)

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'Torch/transformers unavailable')
class CacheBranchTests(unittest.TestCase):
    def test_branch_replication_is_independent_but_gradients_sum(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8s2_cache'),'Safe batch cache missing')
        import torch
        from transformers import Qwen3_5TextConfig
        from transformers.cache_utils import DynamicCache
        from agent.m8s2_cache import fresh_cache,repeat_cache
        cfg=Qwen3_5TextConfig(num_hidden_layers=2,layer_types=['linear_attention','full_attention'])
        cache=fresh_cache(cfg)
        conv=torch.arange(12.,requires_grad=True).reshape(1,3,4);conv.retain_grad()
        recurrent=torch.ones(1,2,3,3,requires_grad=True)
        cache.update_conv_state(conv,0,conv_kernel_size=4)
        cache.update_recurrent_state(recurrent,0)
        k=torch.ones(1,2,5,3,requires_grad=True);v=torch.ones_like(k,requires_grad=True)
        cache.update(k,v,1)
        child=repeat_cache(cache,3)
        self.assertTrue(child.is_recording_past(0),'Single-token convolution must not mutate shared roots')
        self.assertEqual(tuple(child.layers[1].keys.shape),(3,2,5,3))
        loss=child.get_conv_state(0).sum()+child.get_recurrent_state(0).sum()+child.layers[1].keys.sum()
        loss.backward()
        self.assertTrue(torch.equal(conv.grad,torch.full_like(conv,3.)))
        self.assertTrue(torch.equal(recurrent.grad,torch.full_like(recurrent,3.)))
        self.assertTrue(torch.equal(k.grad,torch.full_like(k,3.)))
        with torch.no_grad():child.get_conv_state(0).add_(100)
        self.assertEqual(float(cache.get_conv_state(0)[0,0,0]),0.)
        self.assertEqual(cache.get_seq_length(),5)

if __name__=='__main__':unittest.main()
