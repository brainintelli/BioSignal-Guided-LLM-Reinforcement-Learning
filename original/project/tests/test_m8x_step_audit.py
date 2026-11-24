"""Reward/observation tampering must fail independent step reconstruction."""
import copy
import importlib
import unittest
from agent import m8x_contract as contract
from tests.test_m8x_contract import observation, state


def fixture(assessment=False):
    before=observation(1799,20.,16);after=observation(1802,19.8,16)
    encoded=contract.encode(before)
    v=next(i for i,branch in enumerate(encoded['branches']) if branch)
    action=encoded['branches'][v][0]
    trace=[{**state(after),'tick':tick} for tick in (1800,1801,1802)]
    transition=dict(before=state(before),after=state(after),trace=trace,elapsed_ticks=3)
    nxt=dict(observation=after,transition=transition,terminated=False)
    row=dict(observation=before,next_observation=after,model_input=encoded['messages'],
             target_input=contract.target_messages(encoded,v) if len(encoded['branches'][v])>1 else None,
             verb_index=v,target_index=0,action=action,transition=transition,source='model',salt='m6',
             contract=contract.VERSION,reward=contract.SurvivalReward(before).score(transition,after),
             terminal=False,truncated=assessment)
    if assessment:
        row.update(reward_scope='full_executed_transition',engine_terminated=False,scored_horizon=1800)
    return row,before,nxt


class ExplorationStepAuditTests(unittest.TestCase):
    def module(self):return importlib.import_module('tools.audit_m8x_step')

    def test_training_and_assessment_have_distinct_boundaries(self):
        for assessment in (False,True):
            row,before,nxt=fixture(assessment)
            ended,scored,reward=self.module().validate_step(row,before,nxt,contract.SurvivalReward(before),assessment=assessment)
            self.assertEqual(ended,assessment)
            self.assertEqual(scored['after']['tick'],1800 if assessment else 1802)
            self.assertEqual(reward['elapsed_ticks'],3)

    def test_actual_successor_facts_reward_and_selected_action_are_rebuilt(self):
        for key,value in [('next_observation',observation(1801)),('model_input',['oracle']),
                          ('source','fallback'),('reward',{'reward':100}),('action',{'verb':'drink'}),
                          ('engine_terminated',True),('reward_scope','scored_horizon'),('contract','open_world_balanced_v1')]:
            row,before,nxt=fixture(True);row=copy.deepcopy(row);row[key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,AssertionError)):
                self.module().validate_step(row,before,nxt,contract.SurvivalReward(before),assessment=True)


if __name__=='__main__':unittest.main()
