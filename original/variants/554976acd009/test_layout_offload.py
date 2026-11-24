"""GPU regression for SDPA mask alignment and exact saved-tensor layouts."""
import os,unittest
from contextlib import nullcontext
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
try:
    from layout_offload import saved_on_cpu_preserve_layout,pack,unpack
except ImportError:
    # RED: reproduce the real failure with the initially selected built-in hook.
    saved_on_cpu_preserve_layout=lambda:torch.autograd.graph.save_on_cpu(pin_memory=True)
    pack=unpack=None

class LayoutOffloadTest(unittest.TestCase):
    def setUp(self):
        self.previous_determinism=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous_determinism)
    def test_padded_attention_mask_forward_and_backward_exact(self):
        torch.manual_seed(22554)
        q=torch.randn(1,2,33,64,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,2,902,64,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        bias=torch.zeros((1,1,33,904),device='cuda',dtype=torch.bfloat16)[...,:902]
        self.assertEqual(bias.stride(2),904)
        def run(context):
            inputs=[x.detach().clone().requires_grad_() for x in [q,k,v]]
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),context:
                y=torch.nn.functional.scaled_dot_product_attention(*inputs,attn_mask=bias)
            y.float().sum().backward()
            return y.detach(),[x.grad for x in inputs]
        expected,eg=run(nullcontext());actual,ag=run(saved_on_cpu_preserve_layout())
        self.assertTrue(torch.equal(expected,actual))
        for a,b in zip(eg,ag):self.assertTrue(torch.equal(a,b))

    def test_views_preserve_values_shape_stride(self):
        self.assertIsNotNone(pack)
        x=torch.arange(120,device='cuda',dtype=torch.float32).reshape(10,12)
        for tensor in [x,x.t(),x[1:9,2:10:2],x[:1].expand(6,12),x[:0],x[2,3]]:
            with self.subTest(shape=tuple(tensor.shape),stride=tensor.stride()):
                result=unpack(pack(tensor))
                self.assertEqual(result.shape,tensor.shape);self.assertEqual(result.stride(),tensor.stride())
                self.assertTrue(torch.equal(result,tensor))
        xcpu=torch.arange(12).reshape(3,4).t()
        restored=unpack(pack(xcpu))
        self.assertEqual(restored.stride(),xcpu.stride());self.assertTrue(torch.equal(restored,xcpu))

if __name__=='__main__':unittest.main()
