"""Optimization guards: preserve tokens, cache isolation, and gradient flow."""
import importlib
import importlib.util
import unittest


class FastLayoutTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_fast_layout'),
                             'The validated compact-prefix layout is not implemented')
        return importlib.import_module('agent.m8x_fast_layout')

    def test_prefix_split_preserves_every_suffix_and_order(self):
        split = self.module().split_prefix
        inputs = [[11, 12, 21, 22], [11, 12, 31], [11, 12, 41, 42, 43]]
        self.assertEqual(split(inputs, [11, 12], 5), [[21, 22], [31], [41, 42, 43]])
        self.assertEqual(inputs[0], [11, 12, 21, 22])

    def test_changed_template_and_overflow_are_rejected_without_truncation(self):
        split = self.module().split_prefix
        for inputs,prefix,limit in [([[11,99,21]],[11,12],5), ([[11,12,21]],[11,12],2),
                                     ([[11,12]],[11,12],5), ([],[11],5), ([[11]],[],5)]:
            with self.subTest(inputs=inputs,prefix=prefix), self.assertRaises(ValueError):
                split(inputs,prefix,limit)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'GPU runtime dependencies tested separately on node 5')
class FunctionalCacheTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_fast_policy'))
        return importlib.import_module('agent.m8x_fast_policy')

    def test_cache_branch_does_not_mutate_parent_and_keeps_prefix_gradients(self):
        import torch
        from transformers.cache_utils import DynamicCache
        m=self.module()
        cache=DynamicCache()
        layer=m.FunctionalLinearLayer()
        cache.layers=[layer]
        prefix=torch.tensor([[[1.,2.,3.,4.]]],requires_grad=True)
        state=torch.tensor([[[[2.]]]],requires_grad=True)
        layer.update_conv_state(prefix,conv_kernel_size=4)
        layer.update_recurrent_state(state)
        branch=m.fork_cache(cache)
        full=branch.layers[0].update_conv_state(torch.tensor([[[5.,6.]]]),conv_kernel_size=4)
        previous=branch.layers[0].recurrent_states[0]
        branch.layers[0].update_recurrent_state(previous*3)
        (full.sum()+branch.layers[0].recurrent_states[0].sum()).backward()
        self.assertEqual(cache.layers[0].conv_states[0].tolist(),[[[1.,2.,3.,4.]]])
        self.assertEqual(cache.layers[0].recurrent_states[0].item(),2.)
        self.assertEqual(prefix.grad.tolist(),[[[1.,1.,1.,1.]]])
        self.assertEqual(state.grad.item(),3.)

    def test_optimizer_mutation_changes_revision_without_detaching_parameters(self):
        import torch
        m=self.module()
        model=torch.nn.Linear(2,1,bias=False)
        before=m.parameter_revision(model)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        model(torch.ones(1,2)).sum().backward();optimizer.step()
        self.assertNotEqual(before,m.parameter_revision(model))

    def test_diagnostic_policy_cannot_save_as_registered_release(self):
        m=self.module()
        with self.assertRaisesRegex(RuntimeError,'registration'):
            m.FastPolicy.save(None,'must-not-be-created')


if __name__=='__main__':unittest.main()
