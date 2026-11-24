"""Synthetic CPU evidence only; no fake published acceptance or model sampling."""
import copy
import unittest
from tools.audit_m8x_live_journey import audit, runtime_check
from tests.test_m8x_online import record_fixture as durable_fixture

def fixture():
    server,context,obs=durable_fixture()
    obs.update(world_version=5,day=1,phase='day')
    # Encoding does not consume day/world_version; refresh request digest nonetheless.
    server['observation']=copy.deepcopy(obs)
    from agent.m6_live_session import observation_sha
    server['request_sha256']=observation_sha(obs)
    after=copy.deepcopy(obs);after['tick']=1
    vitals={k:after['player'][k] for k in ('health','hunger','thirst','energy')}
    status=dict(state='running',stop_reason='',damage_conditions=[],death_detail='',model_error='',runtime_day_limit=None,runtime_tick_limit=None)
    client=dict(tick_before=0,tick_after=1,source='model',profile='open3',world_version=5,
        environment_version=obs['environment_version'],balance_sha256=obs['balance_sha256'],
        action=server['decision']['action'],model_decision={k:v for k,v in server['decision'].items()
        if k in ('actor_sha256','decoding','agent_context','rng_seed','rng_before_sha256','rng_after_sha256','raw_output')},
        health=vitals['health'],vitals=vitals,day=1,runtime_status=status)
    class Env:
        def reset(self,*_): return {'observation':copy.deepcopy(obs),'terminated':False}
        def step(self, action):
            assert action==client['action']
            return dict(observation=copy.deepcopy(after),terminated=False,
                transition={'after':dict(vitals,tick=1,dead=False),'trace':[]})
    return client,server,context,Env()

class JourneyTests(unittest.TestCase):
    def test_client_world_fields_and_runtime_events_are_bound(self):
        for key,value in [('world_version',2),('environment_version','legacy'),('balance_sha256','0'*64),('profile','open1')]:
            c,s,ctx,e=fixture();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):audit([c],[s],ctx,e,'a'*64,'b'*64)
        c,s,ctx,e=fixture()
        event=dict(event='runtime_status',tick=1,agent_context=ctx,runtime_status=c['runtime_status'],vitals=c['vitals'],
            world_version=5,environment_version=c['environment_version'],balance_sha256=c['balance_sha256'])
        self.assertEqual(audit([c,event],[s],ctx,e,'a'*64,'b'*64)['runtime_events'],1)
        event['balance_sha256']='0'*64
        with self.assertRaises(ValueError):audit([c,event],[s],ctx,e,'a'*64,'b'*64)

    def test_real_balanced_godot_replay_of_synthetic_transport_records(self):
        from pathlib import Path
        import os
        from agent.m8x_env import GodotEnv
        from agent import m8x_contract as contract
        from agent.m8x_server import decision_identity,target_code
        from agent.m6_live_session import SessionRegistry
        from agent.rl_contract import LABELS
        from tests.m8x_online_fixtures import Stream
        root=Path(__file__).resolve().parents[1]
        godot=os.environ.get('MEADOW_GODOT',str(root/'.tools/godot/Godot_v4.7.2-stable_win64.exe'))
        servers=[];clients=[];sessions=SessionRegistry(Stream,servers.append)
        context=dict(session_id='synthetic-transport-real-godot',episode=0,world_seed=2304)
        with GodotEnv(godot,root) as env:
            obs=env.reset(2304,'open3')['observation']
            for verb_name in ('wait','rest','eat'):
                row=contract.encode(obs);verb=contract.VERBS.index(verb_name);target=0
                action=row['branches'][verb][target]
                def choose(stream):
                    stream.value+=2
                    return dict(decision_identity('a'*64,'b'*64),tick=obs['tick'],source='model',raw_valid=True,
                        action=action,verb_index=verb,target_index=target,model_input=row['messages'],
                        target_input=contract.target_messages(row,verb) if len(row['branches'][verb])>1 else None,
                        verb_probabilities=[float(i==verb) for i in range(len(contract.CODES))],
                        target_probabilities=[float(i==target) for i in range(len(LABELS))],
                        raw_output=contract.CODES[verb]+':'+target_code(row,verb,target))
                decision=sessions.decide(context,obs,choose)
                reply=env.step(action);after=reply['transition']['after']
                clients.append(dict(contract.balance_identity(),tick_before=obs['tick'],tick_after=after['tick'],
                    source='model',action=action,model_decision=decision,health=after['health'],day=reply['observation']['day'],
                    vitals={k:after[k] for k in ('health','hunger','thirst','energy')},
                    runtime_status=dict(state='running',stop_reason='',damage_conditions=[],death_detail='',model_error='',
                        runtime_day_limit=None,runtime_tick_limit=None)))
                obs=reply['observation']
        with GodotEnv(godot,root) as env:
            result=audit(clients,servers,context,env,'a'*64,'b'*64)
        self.assertEqual((result['executed_decisions'],result['verified_ticks']),(3,5))
        self.assertFalse(result['cuda_sampling_reproduced'])
        self.assertFalse(result['complete_episode'])

    def test_prefix_and_pending(self):
        c,s,ctx,e=fixture()
        result=audit([c],[s],ctx,e,'a'*64,'b'*64)
        self.assertEqual(result['verified_ticks'],1)
        self.assertFalse(result['complete_episode'])
        result=audit([], [s],ctx,e,'a'*64,'b'*64)
        self.assertEqual(result['pending_decisions'],1)
        self.assertEqual(result['verified_ticks'],0)
    def test_tampering_and_missing_rows(self):
        for field,value in [('tick_before',1),('tick_after',2),('health',0),('source','human')]:
            c,s,ctx,e=fixture();c[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit([c],[s],ctx,e,'a'*64,'b'*64)
        for which in ('missing','duplicate_client','duplicate_server','rng','observation','runtime'):
            c,s,ctx,e=fixture();clients=[c];servers=[s]
            if which=='missing':servers=[]
            if which=='duplicate_client':clients=[c,c]
            if which=='duplicate_server':servers=[s,s]
            if which=='rng':s['decision']['rng_after_sha256']='0'*64
            if which=='observation':s['observation']['player']['health']=99
            if which=='runtime':c['runtime_status']['runtime_tick_limit']=1800
            with self.subTest(which=which),self.assertRaises(ValueError):audit(clients,servers,ctx,e,'a'*64,'b'*64)
    def test_probe_rejected(self):
        c,s,ctx,e=fixture();ctx['session_id']='m8x-probe-test'
        with self.assertRaises(ValueError):audit([c],[s],ctx,e,'a'*64,'b'*64)

    def test_runtime_death_and_false_death(self):
        c,s,ctx,e=fixture();obs=e.reset()['observation']
        after=dict(c['vitals'],tick=1,dead=False)
        c['runtime_status']['state']='dead'
        with self.assertRaises(ValueError):runtime_check(c,obs,after,[])
        after.update(dead=True,health=0);c['vitals']['health']=0
        c['runtime_status'].update(state='dead',stop_reason='health_depleted',death_detail='饥饿 · 生命耗尽',damage_conditions=['starvation'])
        runtime_check(c,obs,after,[dict(tick=1,damage=1,thirst=1,hunger=0,energy=1)])
        c['runtime_status']['damage_conditions']=[]
        with self.assertRaises(ValueError):runtime_check(c,obs,after,[dict(tick=1,damage=1,thirst=1,hunger=0,energy=1)])

    def test_strict_snapshot_and_unbound_manual(self):
        from pathlib import Path
        import hashlib,tempfile
        from tools.audit_m8x_live_journey import snapshot
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as folder:
            p=Path(folder)/'log.jsonl'
            for raw in (b'{"a":1}',b'{"a":1,"a":2}\n',b'{"a":NaN}\n',b'\n'):
                p.write_bytes(raw)
                with self.assertRaises(ValueError):snapshot(p,hashlib.sha256(raw).hexdigest())
            p.write_bytes(b'{}\n')
            with self.assertRaises(ValueError):snapshot(p,'0'*64)
        c,s,ctx,e=fixture()
        with self.assertRaises(ValueError):audit([{'source':'human'},c],[s],ctx,e,'a'*64,'b'*64)

if __name__=='__main__':unittest.main()
