"""Compare fused BF16 SiLU gating to native value and gradient semantics."""
import importlib
import unittest
import importlib.util
torch=importlib.import_module('torch') if importlib.util.find_spec('torch') else None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
class FusedMLPTests(unittest.TestCase):
    def test_value_and_both_input_gradients(self):
        gate=importlib.import_module('agent.m8x_fused_mlp').SiluGate
        torch.manual_seed(310)
        x=torch.randn(2,127,384,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        up=torch.randn_like(x,requires_grad=True)
        a=x.detach().clone().requires_grad_();b=up.detach().clone().requires_grad_()
        expected=torch.nn.functional.silu(a)*b;actual=gate.apply(x,up)
        torch.testing.assert_close(actual,expected,rtol=.008,atol=.008)
        gradient=torch.randn_like(x)
        actual.backward(gradient);expected.backward(gradient)
        torch.testing.assert_close(x.grad,a.grad,rtol=.008,atol=.008)
        torch.testing.assert_close(up.grad,b.grad,rtol=.008,atol=.008)
        with torch.no_grad():self.assertTrue(torch.equal(gate.apply(x,up),actual))

    def test_split_projection_views_have_the_correct_offsets(self):
        gate=importlib.import_module('agent.m8x_fused_mlp').SiluGate
        x=torch.randn(2,37,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        a,b=x.split(256,dim=-1)
        reference=x.detach().clone().requires_grad_();c,d=reference.split(256,dim=-1)
        got=gate.apply(a,b);expected=torch.nn.functional.silu(c)*d
        gradient=torch.randn_like(got);got.backward(gradient);expected.backward(gradient)
        torch.testing.assert_close(got,expected,rtol=.008,atol=.008)
        torch.testing.assert_close(x.grad,reference.grad,rtol=.008,atol=.008)


if __name__=='__main__':unittest.main()
