"""Guard exact token prefixes and differentiable, isolated cache branches."""
import importlib
import importlib.util
import unittest


class PrefixBoundaryTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_kv_layout'))
        return importlib.import_module('agent.m8x_kv_layout')

    def test_shared_prefix_never_reuses_divergent_assistant_tokens(self):
        split=self.module().checked_suffix
        self.assertEqual(split([10,11,12,21,22],[10,11,12],8),[21,22])
        with self.assertRaises(ValueError):split([10,11,99,21,22],[10,11,12],8)

    def test_empty_single_token_and_overflow_continuations_rejected(self):
        split=self.module().checked_suffix
        for ids,prefix,limit in [([1,2],[1,2],8),([1,2,3],[1,2],8),([1,2,3,4],[1,2],3),([1,2],[],8)]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):split(ids,prefix,limit)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'Node-5 runtime required')
class SharedCacheTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_kv_policy'))
        return importlib.import_module('agent.m8x_kv_policy')

    def test_branches_share_storage_without_mutating_parent_or_losing_gradients(self):
        import torch
        from transformers.cache_utils import DynamicCache,DynamicLayer
        from agent.m8x_fast_policy import FunctionalLinearLayer
        m=self.module()
        cache=DynamicCache()
        linear=FunctionalLinearLayer();attention=DynamicLayer();cache.layers=[linear,attention]
        prefix=torch.tensor([[[1.,2.,3.,4.]]],requires_grad=True)
        state=torch.tensor([[[[2.]]]],requires_grad=True)
        keys=torch.tensor([[[[7.],[8.]]]],requires_grad=True)
        linear.update_conv_state(prefix,conv_kernel_size=4);linear.update_recurrent_state(state)
        attention.update(keys,keys*2)
        left=m.branch_cache(cache);right=m.branch_cache(cache)
        self.assertEqual(left.layers[0].recurrent_states[0].data_ptr(),state.data_ptr())
        self.assertEqual(left.layers[1].keys.data_ptr(),cache.layers[1].keys.data_ptr())
        total=0
        for branch,factor in [(left,3),(right,5)]:
            full=branch.layers[0].update_conv_state(torch.ones(1,1,2),conv_kernel_size=4)
            previous=branch.layers[0].recurrent_states[0]
            branch.layers[0].update_recurrent_state(previous*factor)
            k,v=branch.layers[1].update(torch.ones(1,1,2,1),torch.ones(1,1,2,1))
            total=total+full.sum()+branch.layers[0].recurrent_states[0].sum()+k.sum()+v.sum()
        total.backward()
        self.assertEqual(cache.layers[0].conv_states[0].tolist(),[[[1.,2.,3.,4.]]])
        self.assertEqual(cache.layers[0].recurrent_states[0].item(),2.)
        self.assertEqual(cache.layers[1].keys.shape[-2],2)
        self.assertEqual(prefix.grad.tolist(),[[[2.,2.,2.,2.]]])
        self.assertEqual(state.grad.item(),8.)
        self.assertEqual(keys.grad.tolist(),[[[[6.],[6.]]]])

    def test_unknown_mutating_cache_types_are_rejected(self):
        from transformers.cache_utils import DynamicCache,LinearAttentionLayer
        m=self.module();cache=DynamicCache();cache.layers=[LinearAttentionLayer()]
        with self.assertRaises(TypeError):m.branch_cache(cache)

if __name__=='__main__':unittest.main()
