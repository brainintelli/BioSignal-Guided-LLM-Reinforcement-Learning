import importlib.util,unittest
class MemoryTests(unittest.TestCase):
    def test_update_context_exists_and_preserves_actual_saved_tensor_layout(self):
        spec=importlib.util.find_spec('agent.mea_memory')
        self.assertIsNotNone(spec,'Missing authenticated CPU activation memory policy')
        if importlib.util.find_spec('torch') is None:self.skipTest('GPU host supplies Torch')
        import torch
        from agent.mea_memory import pack,unpack
        if not torch.cuda.is_available():self.skipTest('Actual GPU offload test requires CUDA')
        base=torch.arange(48.,device='cuda',dtype=torch.float64).reshape(6,8)
        for x in [base,base[:,1:7:2],base[1:2,:].expand(5,8),base[:0,:],base[2,3]]:
            saved=pack(x);self.assertEqual(saved[1].device.type,'cpu')
            restored=unpack(saved)
            self.assertEqual(restored.stride(),x.stride());self.assertEqual(restored.dtype,x.dtype)
            self.assertTrue(torch.equal(restored,x))

    def test_trainer_wraps_update_without_altering_gradients_or_budget(self):
        from tools import mea_shuffle_train as t
        self.assertTrue(hasattr(t,'training_update'),'Missing training memory wrapper')
        self.assertEqual((t.CONFIG['microbatch'],t.CONFIG['minibatch'],t.CONFIG['critic_minibatch']),(4,128,128))
        if importlib.util.find_spec('torch') is None:self.skipTest('GPU host supplies Torch')
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual GPU backward test requires CUDA')
        from unittest.mock import patch
        p=torch.nn.Parameter(torch.arange(24.,device='cuda',dtype=torch.float64).reshape(4,6))
        def tiny_update(*args):
            y=(p[:,1:5:2].square()).sum();y.backward();return {'loss':float(y.detach())}
        expected=torch.zeros_like(p);expected[:,1:5:2]=2*p.detach()[:,1:5:2]
        with patch('agent.m8_train.update',tiny_update):result=t.training_update(None,None,None,[],None)
        self.assertGreater(result['loss'],0)
        self.assertTrue(torch.equal(p.grad,expected))
if __name__=='__main__':unittest.main()
