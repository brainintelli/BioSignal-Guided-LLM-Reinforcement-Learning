"""Real CUDA test against the Qwen zero-centered RMSNorm formula."""
import importlib
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'Node-5 runtime required')
class FusedNormTests(unittest.TestCase):
    def test_fused_normalization_preserves_zero_centered_weight_and_backward(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA required')
        self.assertIsNotNone(importlib.util.find_spec('agent.m8x_fused_norm'))
        m=importlib.import_module('agent.m8x_fused_norm')
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        torch.manual_seed(41)
        norm=Qwen3_5RMSNorm(128,eps=1e-6).cuda().requires_grad_(False)
        with torch.no_grad():norm.weight.copy_(torch.linspace(-.4,.4,128,device='cuda'))
        weight=norm.weight.clone()
        x=(torch.randn(1,17,128,device='cuda')*.001).to(torch.bfloat16).requires_grad_()
        expected=norm(x)
        want=torch.autograd.grad(expected.float().square().sum(),x)[0]
        self.assertEqual(m.enable_fused_norm(norm),1)
        actual=norm(x)
        got=torch.autograd.grad(actual.float().square().sum(),x)[0]
        torch.testing.assert_close(actual,expected,rtol=.01,atol=.015)
        torch.testing.assert_close(got,want,rtol=.02,atol=8.)
        with torch.no_grad():plain=norm(x)
        self.assertTrue(torch.equal(plain,actual))
        self.assertTrue(torch.equal(norm.weight,weight))

if __name__=='__main__':unittest.main()
