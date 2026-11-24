"""Run only by root in the authenticated GPU runtime, after real input validation."""
import unittest
from types import SimpleNamespace
from launch import bootstrap_runtime,validate_dataset


def main():
    validate_dataset();bootstrap_runtime()
    import torch
    from sr_common import config,optimizer_hash,steps
    from sr_policy import expand_critic,consistent_values
    from sr_protocol import ARMS

    class GPUAcceptance(unittest.TestCase):
        def test_registered_runtime_and_r1_only_configs(self):
            self.assertEqual(torch.__version__,'2.6.0+cu124');self.assertEqual(torch.cuda.device_count(),1)
            for arm in ARMS:
                c=config(arm,0)
                self.assertEqual(c['r1_weight'],1.);self.assertEqual(c['aux_weight'],0.)
                self.assertFalse(c['r2_enabled']);self.assertFalse(c['aux_enabled'])
        def test_zero_initialized_adapter_preserves_original_function(self):
            torch.manual_seed(10)
            critic=torch.nn.Sequential(torch.nn.Linear(108,128,device='cuda'),torch.nn.Tanh(),torch.nn.Linear(128,1,device='cuda'))
            x=torch.randn(4,108,device='cuda');old=critic(x).detach();saved=critic[0].weight.detach().clone();expand_critic(critic)
            self.assertTrue(torch.equal(critic[0].weight[:,:108],saved));self.assertEqual(torch.count_nonzero(critic[0].weight[:,108:]).item(),0)
            z=torch.cat([x,torch.randn(4,4,device='cuda')],1)
            self.assertTrue(torch.allclose(critic(z),old,atol=1e-6,rtol=0))
            critic(z).sum().backward();self.assertGreater(float(critic[0].weight.grad[:,108:].abs().sum()),0)
        def test_serial_critic_matches_batch_replay(self):
            torch.manual_seed(10)
            critic=torch.nn.Sequential(torch.nn.Linear(112,128,device='cuda'),torch.nn.Tanh(),torch.nn.Linear(128,1,device='cuda'))
            p=SimpleNamespace(critic=critic,device_name='cuda');rows=[dict(critic_input=[float(i%9)/7]*112) for i in range(9)]
            self.assertTrue(torch.equal(consistent_values(p,rows),torch.cat([consistent_values(p,[r]) for r in rows])))
        def test_optimizer_scalar_state_is_hashed(self):
            parameter=torch.nn.Parameter(torch.tensor([1.,2.],device='cuda'));optimizer=torch.optim.AdamW([parameter],lr=.01);opts={'actor':optimizer}
            old=optimizer_hash(opts);parameter.sum().backward();optimizer.step()
            self.assertEqual(steps(opts),{'actor':[1]});self.assertNotEqual(old,optimizer_hash(opts))
            old=optimizer_hash(opts);optimizer.state[parameter]['step'].add_(1);self.assertNotEqual(old,optimizer_hash(opts))

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GPUAcceptance)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
