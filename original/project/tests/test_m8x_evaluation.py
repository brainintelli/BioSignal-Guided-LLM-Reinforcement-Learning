"""Evaluation clips survival time, never invents a successor observation."""
import importlib
import unittest
from agent.m8x_contract import SurvivalReward
from tests.test_m8x_contract import observation, state


class ExplorationEvaluationTests(unittest.TestCase):
    def test_rest_crossing_horizon_keeps_full_reward_and_exact_survival_tick(self):
        m=importlib.import_module('agent.m8x_eval')
        before=observation(1799, 20., 16)
        after=observation(1802, 19.8, 16)
        after['player']['health']=0.
        after['allowed_actions']=[]
        trace=[{**state(before),'tick':1800}, {**state(before),'tick':1801}, state(after,True)]
        transition=dict(before=state(before),after=state(after,True),trace=trace,elapsed_ticks=3)
        reward,scored=m.score_evaluation_transition(SurvivalReward(before),transition,after,1800)
        self.assertEqual(reward['elapsed_ticks'],3)
        self.assertEqual(scored['after']['tick'],1800)
        self.assertFalse(scored['after']['dead'])
        self.assertEqual(transition['after']['tick'],1802)

    def test_death_on_the_horizon_is_not_success(self):
        m=importlib.import_module('agent.m8x_eval')
        before=observation(1799, 20., 16)
        after=observation(1800, 0., 16)
        after['player']['health']=0.
        after['allowed_actions']=[]
        transition=dict(before=state(before),after=state(after,True),trace=[state(after,True)],elapsed_ticks=1)
        reward,scored=m.score_evaluation_transition(SurvivalReward(before),transition,after,1800)
        self.assertTrue(scored['after']['dead'])
        self.assertEqual(reward['elapsed_ticks'],1)


if __name__=='__main__':unittest.main()
