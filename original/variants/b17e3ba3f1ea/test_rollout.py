import unittest
from sr_rollout import finish,validate_sequence

class RolloutTests(unittest.TestCase):
    def test_actual_terminal_boundary_skips_dead_or_reset_valuation(self):
        rows=[dict(reward=1.,value=.3,duration=1,terminal=False,boundary=False),dict(reward=-2.,value=.4,duration=2,terminal=True,boundary=True)]
        def forbid():raise AssertionError('terminal model called')
        boot=finish(rows,forbid);self.assertEqual(boot,dict(value=0.,prepared=None,terminal=True));self.assertAlmostEqual(rows[-1]['return'],-2.)
    def test_interior_terminal_cuts_later_reward_and_tail_uses_cutoff(self):
        def rows(last):return [dict(reward=-1.,value=.3,duration=1,terminal=True,boundary=True),dict(reward=last,value=.4,duration=2,terminal=False,boundary=False)]
        a=rows(2);b=rows(1e6);pa=dict(marker='next_observation_snapshot')
        x=finish(a,lambda:(.9,pa));finish(b,lambda:(.9,pa));self.assertEqual(a[0]['return'],b[0]['return']);self.assertEqual(x['prepared'],pa);self.assertFalse(x['terminal'])
    def test_reset_keeps_neural_index_and_rejects_wrap(self):
        good=[dict(index=0,window=60,seed=10,terminal=True,reset_seed=11),dict(index=1,window=61,seed=11,terminal=False,reset_seed=None)]
        self.assertTrue(validate_sequence(good,[60,61,62],training=True))
        bad=[*good[:1],good[1]|dict(window=60)]
        with self.assertRaises(ValueError):validate_sequence(bad,[60,61,62],training=True)
        with self.assertRaises(ValueError):validate_sequence(good,[60,61,62],training=False)
if __name__=='__main__':unittest.main()
