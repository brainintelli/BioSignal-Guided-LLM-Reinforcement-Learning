"""Falsifiable unit guards; these fixtures are not rollout evidence."""
import copy
from types import SimpleNamespace
import unittest
from tools.audit_m8x_run import validate_step,validate_environment_identity
from agent.m8x_eval import episode_seed
from agent.rl_contract import LABELS


class Tracker:
    def score(self,t):return {'elapsed_ticks':t['elapsed_ticks'],'discount':.999**t['elapsed_ticks'],'total':t['elapsed_ticks']*.01}


def fixture(assessment=False):
    obs={'tick':1799,'profile':'open1'}
    transition={'after':{'tick':1801,'dead':False},'trace':[{'tick':1800,'dead':False},{'tick':1801,'dead':False}],'elapsed_ticks':2}
    encoded={'messages':['facts'],'branches':[[{'verb':'wait'}]],'verb_mask':[True]}
    contract=SimpleNamespace(encode=lambda o,salt:encoded,target_messages=lambda e,v:None)
    row=dict(observation=obs,model_input=['facts'],target_input=None,verb_index=0,target_index=0,action={'verb':'wait'},source='model',transition=transition,
        reward=Tracker().score(transition),terminal=False,truncated=False)
    if assessment:
        row.update(seed=1,replicate=0,rng_seed=episode_seed('m8_exploration_reliable_v1','open1',1,0),scored_horizon=1800,
            verb_probabilities=[1.],target_probabilities=[1.]+[0.]*(len(LABELS)-1),rng_before_sha256='a'*64,rng_after_sha256='b'*64,
            reward=Tracker().score(dict(transition,elapsed_ticks=1)),truncated=True)
    return row,obs,transition,contract


class AuditTests(unittest.TestCase):
    def test_old_world_cannot_pass_as_balanced(self):
        reg={'balance_sha256':'a'*64}
        value=dict(arm='candidate',profile='open3',world_version=5,environment_version='m8_exploration_v1',balance_sha256='a'*64)
        validate_environment_identity(value,reg)
        for field,bad in [('world_version',2),('environment_version',None),('balance_sha256','b'*64),('profile','open1')]:
            with self.subTest(field=field),self.assertRaises(ValueError):validate_environment_identity(dict(value,**{field:bad}),reg)
        old=dict(arm='original_v2',profile='open1',world_version=2,environment_version=None,balance_sha256=None)
        validate_environment_identity(old,reg)
        del old['environment_version']
        with self.assertRaises(ValueError):validate_environment_identity(old,reg)

    def test_training_is_not_truncated_at_1800(self):
        row,obs,t,c=fixture();ended,scored,_=validate_step(row,obs,t,Tracker(),c)
        self.assertFalse(ended);self.assertEqual(scored['after']['tick'],1801)
        row['truncated']=True
        with self.assertRaises(AssertionError):validate_step(row,obs,t,Tracker(),c)

    def test_assessment_clips_exact_tick_and_checks_full_transition(self):
        row,obs,t,c=fixture(True)
        ended,scored,_=validate_step(row,obs,t,Tracker(),c,assessment=True,rng_prefix='m8_exploration_reliable_v1')
        self.assertTrue(ended);self.assertEqual(scored['after']['tick'],1800)
        row=copy.deepcopy(row);row['transition']['after']['tick']=1800
        with self.assertRaises(AssertionError):validate_step(row,obs,t,Tracker(),c,assessment=True,rng_prefix='m8_exploration_reliable_v1')

    def test_inputs_actions_reward_discount_and_rng_are_not_trusted(self):
        changes=[('model_input',['oracle']),('target_input',['route']),('action',{'verb':'drink'}),('source','fallback'),
            ('reward',{'elapsed_ticks':1,'discount':1.,'total':.01}),('rng_seed',0),('target_probabilities',[1.]*len(LABELS)),('rng_after_sha256','x'*64)]
        for key,value in changes:
            with self.subTest(key=key):
                row,obs,t,c=fixture(True);row[key]=value
                with self.assertRaises((AssertionError,ValueError)):
                    validate_step(row,obs,t,Tracker(),c,assessment=True,rng_prefix='m8_exploration_reliable_v1')


if __name__=='__main__':unittest.main()
