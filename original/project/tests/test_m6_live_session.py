"""Transport and release fixtures, not model-quality evidence."""
import copy
import hashlib
import unittest

from agent.m6_live_session import SessionRegistry, live_seed, validate_release, STOCHASTIC


class Generator:
    def __init__(self, seed): self.state = str(seed).encode()
    def snapshot(self): return self.state
    def restore(self, state): self.state = state
    def advance(self): self.state = hashlib.sha256(self.state).digest()


class LiveSessionTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.registry = SessionRegistry(Generator, self.records.append)
        self.context = dict(session_id="fixture-only", episode=1, world_seed=2304)
        self.obs = dict(tick=0, facts="same observation")
        self.calls = 0

    def sample(self, generator):
        self.calls += 1
        generator.advance(); generator.advance()
        return dict(tick=self.obs['tick'], source='model', raw_valid=True, action={'verb':'wait'})

    def test_retry_never_resamples(self):
        first = self.registry.decide(self.context, self.obs, self.sample)
        first['action']['verb'] = 'corrupted caller copy'
        retry = self.registry.decide(self.context, self.obs, self.sample)
        self.assertEqual(retry['action']['verb'], 'wait')
        self.assertEqual(self.calls, 1); self.assertEqual(len(self.records), 1)

    def test_changed_same_tick_and_older_tick_are_rejected(self):
        self.registry.decide(self.context, self.obs, self.sample)
        with self.assertRaises(ValueError):
            self.registry.decide(self.context, dict(self.obs,facts='changed'), self.sample)
        self.obs['tick'] = 3
        self.registry.decide(self.context, self.obs, self.sample)
        with self.assertRaises(ValueError):
            self.registry.decide(self.context, dict(self.obs,tick=0), self.sample)
        self.assertEqual(self.calls, 2)

    def test_failure_restores_rng_and_does_not_commit(self):
        states = []
        def fail(g):
            states.append(g.snapshot()); g.advance(); raise RuntimeError('inference failed')
        for _ in range(2):
            with self.assertRaises(RuntimeError): self.registry.decide(self.context,self.obs,fail)
        self.assertEqual(states[0],states[1]); self.assertEqual(self.records,[])

    def test_commit_failure_restores_rng(self):
        def fail(_): raise OSError('disk full')
        reg = SessionRegistry(Generator, fail)
        with self.assertRaises(OSError): reg.decide(self.context,self.obs,self.sample)
        reg.commit = self.records.append
        with self.assertRaises(RuntimeError): reg.decide(self.context,self.obs,self.sample)
        self.assertEqual(self.records,[])
        self.assertEqual(reg.sessions[('fixture-only',1)]['generator'].snapshot(),Generator(live_seed(self.context)).snapshot())

    def test_restart_replays_cache_and_next_rng_state(self):
        first=self.registry.decide(self.context,self.obs,self.sample)
        restarted=SessionRegistry(Generator,self.records.append)
        restarted.restore_records(copy.deepcopy(self.records))
        self.assertEqual(restarted.decide(self.context,self.obs,self.sample),first)
        self.assertEqual(self.calls,1)
        self.obs['tick']=1
        restarted.decide(self.context,self.obs,self.sample)
        a=self.records[-1]['generator_state_b64']
        self.registry.decide(self.context,self.obs,self.sample)
        self.assertEqual(a,self.records[-1]['generator_state_b64'])

    def test_world_change_within_same_episode_is_rejected(self):
        self.registry.decide(self.context,self.obs,self.sample)
        with self.assertRaises(ValueError):
            self.registry.decide(dict(self.context,world_seed=1),self.obs,self.sample)

    def test_seed_is_stable_and_separate_from_actor_input(self):
        self.assertEqual(live_seed(self.context),live_seed(dict(reversed(list(self.context.items())))))
        self.assertNotEqual(live_seed(self.context),live_seed(dict(self.context,episode=2)))
        decision=self.registry.decide(self.context,self.obs,self.sample)
        self.assertEqual(decision['rng_seed'],str(live_seed(self.context)), 'JSON transport must not lose 63-bit seed precision in Godot')
        self.assertEqual(self.records[0]['observation'],self.obs)
        self.assertNotIn('agent_context',self.obs)

    def test_registry_does_not_silently_evict_rng_streams(self):
        reg=SessionRegistry(Generator,self.records.append,max_sessions=1)
        reg.decide(self.context,self.obs,self.sample)
        with self.assertRaises(ValueError): reg.decide(dict(self.context,episode=2),self.obs,self.sample)

    def test_invalid_context_and_tick_rejected(self):
        for bad in (dict(self.context,episode=True),dict(self.context,session_id=''),dict(self.context,world_seed=1.5)):
            with self.assertRaises(ValueError): self.registry.decide(bad,self.obs,self.sample)
        with self.assertRaises(ValueError): self.registry.decide(self.context,dict(self.obs,tick=True),self.sample)


class ReleaseTests(unittest.TestCase):
    def report(self):
        return dict(protocol='m6_stochastic_v1',gate_passed=True,overall_status='evaluated',
            candidate_actor_sha256='a'*64,primary_gate_passed=True,
            primary=dict(gate_passed=True,candidate_decoding=STOCHASTIC),
            required_audits={k:dict(passed=True,sha256='b'*64,trace_sha256='c'*64,replayed_decisions=80)
                for k in ('base_replay_audit.json','candidate_replay_audit.json','rng_base.json','rng_candidate.json')},
            historical_audits={arm+'_replay_audit.json':dict(passed=True,sha256='b'*64,actor_sha256='d'*64,
                report_sha256='b'*64,episodes=20,replayed_decisions=80) for arm in ('legacy_base','legacy_checkpoint')},
            report_sha256={k:'b'*64 for k in ('base_replay_audit.json','candidate_replay_audit.json','rng_base.json','rng_candidate.json',
                'legacy_base_replay_audit.json','legacy_checkpoint_replay_audit.json','legacy_base.json','legacy_checkpoint.json')},
            historical_guards={k:dict(gate_passed=True,before_actor_sha256='d'*64) for k in ('legacy_fresh_base','legacy_checkpoint0016')})

    def test_stochastic_result_never_authorizes_greedy(self):
        validate_release(self.report(),'a'*64,'stochastic')
        with self.assertRaises(ValueError): validate_release(self.report(),'a'*64,'greedy')

    def test_missing_guard_or_actor_mismatch_rejects(self):
        report=self.report();report['historical_guards'].pop('legacy_fresh_base')
        with self.assertRaises(ValueError): validate_release(report,'a'*64,'stochastic')
        with self.assertRaises(ValueError): validate_release(self.report(),'b'*64,'stochastic')

    def test_unverified_or_mismatching_audits_reject(self):
        report=self.report();report.pop('required_audits')
        with self.assertRaises(ValueError): validate_release(report,'a'*64,'stochastic')
        for key,value in [('passed',False),('sha256','d'*64),('trace_sha256','x'),('replayed_decisions',0)]:
            report=self.report();report['required_audits']['rng_base.json'][key]=value
            with self.assertRaises(ValueError): validate_release(report,'a'*64,'stochastic')

    def test_missing_or_wrong_historical_audit_rejects(self):
        report=self.report();report.pop('historical_audits')
        with self.assertRaises(ValueError): validate_release(report,'a'*64,'stochastic')
        for key,value in [('passed',False),('actor_sha256','e'*64),('report_sha256','f'*64),('episodes',19)]:
            report=self.report();report['historical_audits']['legacy_base_replay_audit.json'][key]=value
            with self.assertRaises(ValueError): validate_release(report,'a'*64,'stochastic')


if __name__ == '__main__': unittest.main()
