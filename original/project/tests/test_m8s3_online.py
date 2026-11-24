"""Synthetic contract tests; never evidence of real model survival or CUDA behavior."""
import copy
import importlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager


class SurvivalOnlineGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path(__file__).resolve().parents[1]/'output/runtime_tests').mkdir(parents=True,exist_ok=True)

    def api(self):
        return importlib.import_module('agent.m8s3_online_gate')

    def reports(self, successes=100):
        rows=[dict(seed=2000+i,replicate=0,survival_ticks=1800 if i<successes else 1000,
                   terminal=i>=successes,alive_at_horizon=i<successes) for i in range(100)]
        baseline=[dict(seed=2000+i,replicate=0,survival_ticks=900,terminal=True,alive_at_horizon=False) for i in range(100)]
        c=dict(test_seeds=list(range(2000,2100)),profile='open4',world_version=6,temperature=1)
        candidate=dict(episodes=rows,actor_sha256='candidate',profile='open4',world_version=6,temperature=1,reference=False)
        parent=dict(episodes=baseline,actor_sha256='parent',profile='open4',world_version=6,temperature=1,reference=True)
        old=dict(episodes=baseline,actor_sha256='parent',profile='open1',temperature=1)
        return c,candidate,parent,old

    def test_recomputes_numeric_gate_from_complete_registered_worlds(self):
        api=self.api();c,a,b,d=self.reports(90)
        result=api.verify_test_reports(c,a,b,d,'candidate','parent')
        self.assertTrue(result['numerical_gate_passed'])
        a['episodes'][89].update(survival_ticks=1000,terminal=True,alive_at_horizon=False)
        a['summary']={'successes':100}
        with self.assertRaisesRegex(ValueError,'numerical'):
            api.verify_test_reports(c,a,b,d,'candidate','parent')

    def test_wrong_world_actor_and_policy_cannot_authorize_online(self):
        api=self.api();c,a,b,d=self.reports()
        for field,value in [('actor_sha256','other'),('temperature',.5),('profile','open3'),('reference',True)]:
            altered=copy.deepcopy(a);altered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                api.verify_test_reports(c,altered,b,d,'candidate','parent')
        a['episodes'][0]['seed']=9000
        with self.assertRaisesRegex(ValueError,'world matrix'):
            api.verify_test_reports(c,a,b,d,'candidate','parent')

    def test_missing_independent_evidence_never_authorizes_a_server(self):
        api=self.api()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            with self.assertRaises((ValueError,FileNotFoundError)):
                api.verify_offline_gate(Path(tmp))

    def test_rebuilds_report_endpoints_from_audited_trace(self):
        api=self.api()
        row=dict(seed=20,actor_sha256='actor',transition={'trace':[{'tick':1800,'dead':True}]})
        report={'episodes':[dict(seed=20,replicate=0,decisions=1,survival_ticks=1800,terminal=True,alive_at_horizon=False)]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as tmp:
            path=Path(tmp)/'trace.jsonl.gz'
            with gzip.open(path,'wt') as f: f.write(json.dumps(row)+'\n')
            api.verify_report_endpoints(path,report,'actor')
            report['episodes'][0].update(terminal=False,alive_at_horizon=True)
            with self.assertRaisesRegex(ValueError,'endpoint'):
                api.verify_report_endpoints(path,report,'actor')


class SurvivalLiveProbeTests(unittest.TestCase):
    def test_actual_game_chain_rejects_manual_and_changed_actions(self):
        api=importlib.import_module('tools.audit_m8s3_online')
        from tests.test_m8s3_contract import observation
        obs=observation();obs['tick']=0;next_obs=copy.deepcopy(obs);next_obs['tick']=1
        context={'session_id':'synthetic','episode':0,'world_seed':123}
        d=dict(agent_context=context,actor_sha256='actor',registration_sha256='reg',action={'verb':'wait'})
        server={'observation':obs,'decision':d,'agent_context':context}
        after=dict(tick=1,dead=False,**{k:next_obs['player'][k] for k in ('health','hunger','thirst')})
        client=dict(tick_before=0,tick_after=1,source='model',profile='open4',world_version=6,
                    action={'verb':'wait'},model_decision={k:v for k,v in d.items() if k!='action'},
                    vitals={k:after[k] for k in ('health','hunger','thirst')},
                    runtime_status={'runtime_day_limit':None,'runtime_tick_limit':None,'state':'running','stop_reason':'','model_error':''})
        class Env:
            def reset(self,*_):return {'observation':obs}
            def step(self,_):return {'observation':next_obs,'transition':{'after':after}}
        result=api.verify_client_steps([client],[server],Env(),context)
        self.assertEqual(result['executed_decisions'],1)
        for patch in ({'source':'human'},{'action':{'verb':'rest'}},{'tick_after':5}):
            with self.assertRaises((ValueError,AssertionError)):
                api.verify_client_steps([dict(client,**patch)],[server],Env(),context)

    def test_fixed_training_pairs_reject_heldout_states_and_discontinuity(self):
        api=importlib.import_module('tools.m8s3_live_sampler_probe')
        c=dict(train_seed_start=100,train_seed_count=10,validation_seeds=[200],test_seeds=[300])
        rows=[dict(update=1,env=i,seed=100+i,observation={'tick':j},next_observation={'tick':j+1},
                   transition={'after':{'dead':False}}) for j in range(2) for i in range(4)]
        self.assertEqual(len(api.select_pairs(iter(rows),c)),4)
        changed=copy.deepcopy(rows);changed[0]['seed']=300
        with self.assertRaises(ValueError):api.select_pairs(iter(changed),c)
        changed=copy.deepcopy(rows);changed[4]['observation']['tick']=5
        with self.assertRaises(ValueError):api.select_pairs(iter(changed),c)

    def test_http_health_requires_exact_offline_accepted_identity(self):
        api=importlib.import_module('tools.check_m8s3_live_link')
        from agent.m8s3_server import decision_identity
        identity=decision_identity('a'*64,'b'*64,.1)
        h=dict(identity,decision_ready=True,validation_authorized=True,numerical_gate_passed=True)
        api.validate_health(h,identity)
        for patch in ({'validation_authorized':False},{'actor_sha256':'c'*64},{'temperature':.5}):
            with self.assertRaises(ValueError):api.validate_health(dict(h,**patch),identity)


class SurvivalOnlineSamplingTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('agent.m8s3_server')

    def test_two_draws_share_canonical_scope_even_for_singleton_target(self):
        from tests.m8x_online_fixtures import FakePolicy,FakeTorch,Stream
        class Policy(FakePolicy):
            inside=False
            @contextmanager
            def _scope(self,prepared,reference):
                self.inside=True
                try: yield
                finally: self.inside=False
            def target_distribution(self,prepared,verbs):
                if not self.inside: raise RuntimeError('Root scope was lost')
                return super().target_distribution(prepared,verbs)
        policy=Policy();torch=FakeTorch();stream=Stream(0)
        prepared=[dict(verb_mask=[True],branches=[[{'verb':'wait'}]])]
        result=self.api().sample_stochastic(policy,prepared,stream,torch)
        self.assertEqual(result['action'],{'verb':'wait'})
        self.assertEqual(len(torch.draws),2)
        self.assertFalse(policy.inside)

    def test_durable_semantic_record_binds_world_actor_and_exploration(self):
        from tests.test_m8s3_contract import observation
        from agent.m8s3_contract import encode
        api=self.api();obs=observation();obs['schema_version']=1;row=encode(obs)
        v=next(i for i,b in enumerate(row['branches']) if b);t=0
        identity=api.decision_identity('a'*64,'b'*64,.1)
        d=dict(identity,tick=obs['tick'],source='model',raw_valid=True,
               action=row['branches'][v][t],verb_index=v,target_index=t,
               model_input=row['messages'],raw_output=api.semantic_output(row,v,t),
               verb_probabilities=[float(i==v) for i in range(8)],
               target_probabilities=[1.]+[0.]*63)
        record={'observation':obs,'decision':d}
        api.validate_record(record,identity)
        for field,value in [('actor_sha256','c'*64),('verb_exploration_epsilon',0.),('world_version',5),('model_input',[])]:
            altered=copy.deepcopy(record);altered['decision'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                api.validate_record(altered,identity)


if __name__=='__main__':
    unittest.main()
