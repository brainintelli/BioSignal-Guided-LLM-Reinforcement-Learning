import unittest,torch
from sr_common import optimizer_hash,steps,config
class RuntimeTests(unittest.TestCase):
    def test_scalar_optimizer_state_identity_and_continuity(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.],device='cuda'));a=torch.optim.AdamW([p],lr=.01);opts={'actor':a}
        self.assertEqual(steps(opts),{'actor':[]});h=optimizer_hash(opts);p.sum().backward();a.step()
        self.assertEqual(steps(opts),{'actor':[1]});self.assertNotEqual(h,optimizer_hash(opts));first=optimizer_hash(opts)
        a.state[p]['step'].add_(1);self.assertNotEqual(first,optimizer_hash(opts))
    def test_arm_configuration_has_explicit_additive_reward(self):
        for arm in ['baseline','state','reward','both','shuffled']:
            c=config(arm,3);self.assertNotIn('alpha',c);self.assertEqual(c['r1_weight'],1.)
            self.assertEqual(c['r2_enabled'],arm in ['reward','both','shuffled']);self.assertEqual(c['state2_enabled'],arm in ['state','both','shuffled'])
if __name__=='__main__':unittest.main()
