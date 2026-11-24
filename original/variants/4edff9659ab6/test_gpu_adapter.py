"""Executed only in the authenticated GPU runtime; no test stubs for training."""
import unittest,torch
from types import SimpleNamespace
from sr_policy import expand_critic,consistent_values
class AdapterTests(unittest.TestCase):
    def test_expansion_preserves_old_function_and_only_new_columns_train(self):
        torch.manual_seed(10);critic=torch.nn.Sequential(torch.nn.Linear(108,128,device='cuda'),torch.nn.Tanh(),torch.nn.Linear(128,1,device='cuda'))
        x=torch.randn(4,108,device='cuda');old=critic(x).detach();saved=critic[0].weight.detach().clone();expand_critic(critic)
        self.assertTrue(torch.equal(critic[0].weight[:,:108],saved));self.assertEqual(torch.count_nonzero(critic[0].weight[:,108:]).item(),0)
        z=torch.cat([x,torch.randn(4,4,device='cuda')],1);self.assertTrue(torch.allclose(critic(z),old,atol=1e-6,rtol=0))
        critic(z).sum().backward();self.assertGreater(float(critic[0].weight.grad[:,108:].abs().sum()),0)
    def test_serial_and_batched_critic_use_identical_reduction(self):
        torch.manual_seed(10);critic=torch.nn.Sequential(torch.nn.Linear(112,128,device='cuda'),torch.nn.Tanh(),torch.nn.Linear(128,1,device='cuda'))
        p=SimpleNamespace(critic=critic,device_name='cuda');rows=[dict(critic_input=[float(i%9)/7]*112) for i in range(9)]
        a=consistent_values(p,rows);b=torch.cat([consistent_values(p,[r]) for r in rows]);self.assertTrue(torch.equal(a,b))
        a.sum().backward();self.assertGreater(float(critic[0].weight.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()
