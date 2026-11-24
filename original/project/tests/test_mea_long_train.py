import copy
import importlib
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile


class LongTrainingTests(unittest.TestCase):
    def module(self):return importlib.import_module('tools.mea_long_train')

    def test_fixed_split_rejects_short_or_reused_source(self):
        m=self.module()
        self.assertEqual(m.window_split(1797)['train_current'],[0,1024])
        self.assertEqual(m.window_split(1686)['heldout'],[1085,1686])
        for count in [1685,0,True,1700.5]:
            with self.assertRaises(ValueError):m.window_split(count)

    def test_terminal_cuts_recursion_before_reset_and_cutoff_bootstraps(self):
        m=self.module()
        rows=[dict(reward=1.,value=.3,duration=2,terminal=False),
              dict(reward=-2.,value=.2,duration=3,terminal=True),
              dict(reward=4.,value=.7,duration=1,terminal=False)]
        m.finish_trajectory(rows,8.,.9,.95)
        self.assertEqual(rows[1]['next_value'],0.)
        self.assertAlmostEqual(rows[1]['advantage'],-2.2)
        self.assertAlmostEqual(rows[0]['advantage'],1+.9**2*.2-.3+(.9*.95)**2*(-2.2))
        self.assertAlmostEqual(rows[2]['return'],4+.9*8)
        changed=copy.deepcopy(rows);changed[2]['reward']=1e6
        m.finish_trajectory(changed,8.,.9,.95)
        self.assertEqual([r['return'] for r in rows[:2]],[r['return'] for r in changed[:2]])
        m.audit_gae(rows,8.,.9,.95)
        rows[0]['next_value']+=1
        with self.assertRaises(ValueError):m.audit_gae(rows,8.,.9,.95)

    def test_optimizer_continuity_rejects_reset_and_partial_state(self):
        m=self.module()
        m.optimizer_gate({'actor':[2,2],'critic':[16,16]}, {'actor':[4,4],'critic':[32,32]},2,16)
        for after in [{'actor':[2,2],'critic':[16,16]}, {'actor':[4],'critic':[32,32]}, {'actor':[4,3],'critic':[32,32]}]:
            with self.assertRaises(ValueError):m.optimizer_gate({'actor':[2,2],'critic':[16,16]},after,2,16)

    def test_first_checkpoint_must_be_original_with_empty_optimizer(self):
        m=self.module();initial=dict(actor='original',critic='critic',base='base',reference='ref')
        manifest=dict(update=0,registration_sha256='pin',actor_sha256='original')
        opt={k:{'state':{}} for k in ('actor','critic')}
        m.validate_initial_checkpoint(initial,initial,manifest,opt,'pin')
        for loaded,cm,stored in [(initial|{'actor':'other'},manifest,opt),
                (initial,manifest|{'update':1},opt),(initial,manifest|{'registration_sha256':'other'},opt),
                (initial,manifest,{'actor':{'state':{0:{'step':1}}},'critic':{'state':{}}})]:
            with self.assertRaises(ValueError):m.validate_initial_checkpoint(initial,loaded,cm,stored,'pin')

    def test_round_report_must_match_independent_evidence(self):
        m=self.module();before=dict(actor='old',critic='oldc',base='b',reference='r')
        after=dict(actor='new',critic='newc',base='b',reference='r')
        rows=[dict(index=i,terminal=i==3,duration=2,prepared={'input_ids':[1,2,3]}) for i in range(128)]
        sr=dict(decisions=128,ticks=256,terminal_indices=[3],maximum_input_tokens=3,
            before=before,after=after,checks={k:True for k in m.CHECK_NAMES},
            applied_steps={'actor':2,'critic':16},metrics={'optimizer_steps':2},
            pre_update_statistic_error=0.,roundtrip_error=0.)
        m.validate_round_evidence(sr,before,after,before,rows,0.,True,True)
        mutations=[{'checks':{}},{'applied_steps':{'actor':3,'critic':16}},
            {'applied_steps':{'actor':2,'critic':15}},{'metrics':{'optimizer_steps':1}},
            {'ticks':999},{'terminal_indices':[]},{'maximum_input_tokens':99},
            {'pre_update_statistic_error':.1},{'roundtrip_error':float('nan')}]
        for change in mutations:
            with self.subTest(change=change),self.assertRaises(ValueError):m.validate_round_evidence(sr|change,before,after,before,rows,0.,True,True)
        for bad_after in [after|{'actor':'old'},after|{'critic':'oldc'},after|{'base':'changed'},after|{'reference':'changed'}]:
            with self.assertRaises(ValueError):m.validate_round_evidence(sr|{'after':bad_after},before,bad_after,before,rows,0.,True,True)
        with self.assertRaises(ValueError):m.validate_round_evidence(sr,before,after,before,rows,0.,False,True)

    def test_invalid_rows_cannot_become_training(self):
        m=self.module()
        for bad in [dict(reward=1.,value=0.,duration=0,terminal=False),
                    dict(reward=float('nan'),value=0.,duration=1,terminal=False),
                    dict(reward=1.,value=0.,duration=1,terminal=1)]:
            with self.assertRaises(ValueError):m.finish_trajectory([bad],0.,.9,.95)

    def test_rollout_reset_keeps_neural_index_and_bootstrap_input(self):
        from tests.test_mea_replay import fixture
        from tests.test_mea_p0 import observation
        from agent.mea_replay import AuditedRecording
        from agent.mea_contract import encode
        m=self.module()
        class Tracker:
            def __init__(self,*args):pass
            def score(self,*args):return dict(reward=.5,elapsed_ticks=1)
        class Env:
            n=0
            def reset(self,seed):
                o=observation();o['tick']=0
                return dict(observation=o)
            def step(self,action):
                self.n+=1;o=observation();o['tick']=self.n
                return dict(observation=o,transition={'step':self.n},terminated=self.n==1)
        class Policy:
            def prepare(self,observations,snapshots):return [encode(o,s) for o,s in zip(observations,snapshots)]
            def act(self,prepared,with_reference):
                p=prepared[0];v=next(i for i,b in enumerate(p['branches']) if b)
                return dict(actions=[p['branches'][v][0]],verbs=[v],targets=[0],values=[.25],logp=[-.5],
                    verb_logp=[-.2],reference_verb=[[1.]],reference_target=[[1.]])
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'Reward',Tracker),patch.object(m,'_value',return_value=.75):
            root=Path(tmp);pin=fixture(root);data=AuditedRecording(root,manifest_sha256=pin);env=Env()
            rows,boot,obs,tracker,seed=m.collect_rollout(Policy(),data,env,env.reset(10)['observation'],Tracker(),10,0,2)
            self.assertEqual([r['snapshot']['index'] for r in rows],[0,1]);self.assertEqual(seed,11)
            self.assertIsNone(rows[0]['next_snapshot']);self.assertEqual(rows[0]['next_value'],0.)
            self.assertEqual(rows[1]['observation'],rows[0]['reset']['observation'])
            self.assertEqual(boot['snapshot'],rows[1]['next_snapshot']);self.assertEqual(boot['snapshot']['index'],2)
            self.assertEqual(boot['value'],.75);self.assertEqual(rows[1]['next_value'],.75)
            self.assertTrue(all(r['reward']==r['r1_score']['reward'] for r in rows))
            self.assertEqual(rows[0]['reward_record']['neural_metrics']['spike_count'],2.)

if __name__=='__main__':unittest.main()
