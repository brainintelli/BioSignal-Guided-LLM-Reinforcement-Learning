import copy, unittest
from protocol import audit_gae, resource_gate, optimizer_gate

class ProtocolTests(unittest.TestCase):
    def rows(self):
        # Terminal ends episode 0; episode 1 is a nonterminal cutoff.
        return [dict(reward=-2.,value=3.,duration=2,terminal=True,boundary=True,advantage=-5.,**{'return':-2.}),
                dict(reward=4.,value=5.,duration=1,terminal=False,boundary=True,advantage=.8,**{'return':5.8})]
    def test_terminal_isolates_later_episode(self):
        r=self.rows();a=audit_gae(r,2.,.9,.8)
        r[1].update(reward=100.,advantage=96.8,**{'return':101.8})
        b=audit_gae(r,2.,.9,.8)
        self.assertEqual(a['advantages'][0],b['advantages'][0])
    def test_fake_bootstrap_rejected(self):
        r=self.rows();r[0]['advantage']=-3.
        with self.assertRaises(AssertionError):audit_gae(r,2.,.9,.8)
    def test_missing_terminal_boundary_rejected(self):
        r=self.rows();r[0]['boundary']=False
        with self.assertRaises(AssertionError):audit_gae(r,2.,.9,.8)
    def test_optimizer_persists(self):
        optimizer_gate({'actor':[8,8],'critic':[64,64]}, {'actor':[11,11],'critic':[128,128]},3,64)
        with self.assertRaises(AssertionError):optimizer_gate({'actor':[8,8],'critic':[64,64]}, {'actor':[3,3],'critic':[64,64]},3,64)
    def test_resource_thresholds(self):
        r=[{'allocated':10,'rss':10},{'allocated':20,'rss':20},{'allocated':21,'rss':21}]
        self.assertTrue(resource_gate(r)['passed'])
        r[-1]['allocated']=20+512*1024**2+1
        self.assertFalse(resource_gate(r)['passed'])
if __name__=='__main__':unittest.main()
