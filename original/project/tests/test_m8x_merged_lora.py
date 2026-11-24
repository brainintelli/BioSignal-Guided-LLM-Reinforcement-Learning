"""Independent autograd reference and actual optimizer cache invalidation."""
import importlib
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'Node-5 torch runtime required')
class MergedLoRATests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_merged_lora'))
        return importlib.import_module('agent.m8x_merged_lora')

    def test_values_and_all_gradients_match_independent_dense_formula(self):
        import torch
        m=self.module();torch.manual_seed(91)
        x=torch.randn(2,5,3,dtype=torch.float64,requires_grad=True)
        a=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
        b=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
        base=torch.randn(4,3,dtype=torch.float64);bias=torch.randn(4,dtype=torch.float64)
        cache=m.MergedWeight()
        actual=m.MergedLinear.apply(x,a,b,cache.get(base,a,b,.7),bias,.7)
        weight=base+.7*(b@a)
        expected=torch.nn.functional.linear(x,weight,bias)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        ga=torch.autograd.grad(actual.square().sum(),(x,a,b))
        ge=torch.autograd.grad(expected.square().sum(),(x,a,b))
        for got,want in zip(ga,ge):torch.testing.assert_close(got,want,rtol=1e-10,atol=1e-10)

    def test_optimizer_step_refreshes_weight_without_mutating_base(self):
        import torch
        m=self.module()
        base=torch.ones(3,2);before=base.clone()
        a=torch.nn.Parameter(torch.ones(1,2));b=torch.nn.Parameter(torch.ones(3,1))
        cache=m.MergedWeight();first=cache.get(base,a,b,2.)
        self.assertEqual(cache.get(base,a,b,2.).data_ptr(),first.data_ptr())
        optimizer=torch.optim.AdamW([a,b],lr=.01)
        y=m.MergedLinear.apply(torch.ones(1,2),a,b,first,None,2.)
        y.square().mean().backward();optimizer.step()
        second=cache.get(base,a,b,2.)
        self.assertFalse(torch.equal(first,second))
        torch.testing.assert_close(second,base+2*(b@a))
        torch.testing.assert_close(base,before)

if __name__=='__main__':unittest.main()
