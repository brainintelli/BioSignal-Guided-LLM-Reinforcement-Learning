"""Offline runtime tests: synthetic policy only; never a Qwen/GPU claim."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def observation():
    from tests.test_mea_p0 import observation as fixture
    return fixture()


def snapshot(obs=None):
    from agent.m9_contract import encode
    s = dict(schema='mea.online.snapshot.v1', binding_sha256='b'*64,
             request_id='req-private', window_sha256='a'*64, source='simulation',
             context_id=digest(encode(obs or observation(), 'reserves')),
             start_frame=123400, end_frame=143400, feature_values=[14, 7., .5, 23.5],
             normalized_values=[1., -2., .25, 3.], normalization_id='n'*64,
             coverage_basis='filtered_transport_frames')
    s['snapshot_id'] = digest(s)
    return s


def config():
    return dict(source='simulation', binding_sha256='b'*64, normalization_id='n'*64,
                devices=[0], batch_size=16, seed=71, r2_enabled=False, stimulation=False,
                model_paths=dict(model='model', initial_checkpoint='initial',
                                 parent_checkpoint='parent', warm_checkpoint='warm'))


class ProjectionTests(unittest.TestCase):
    def runtime(self):
        try:
            return importlib.import_module('mea_online.m9_runtime')
        except ModuleNotFoundError as e:
            self.fail('real online runtime not implemented: ' + str(e))

    def test_projection_excludes_identity_and_has_exact_four_extra_features(self):
        m = self.runtime(); obs = observation(); s = snapshot(obs)
        row = m.project_observation(obs, s, config())
        text = json.dumps(row['messages'], ensure_ascii=False)
        for private in ['req-private', '123400', '143400', 'simulation', 'b'*64, 'a'*64, 'n'*64]:
            self.assertNotIn(private, text)
        for value in ['14', '7.0', '0.5', '23.5']:
            self.assertIn(value, text)
        self.assertEqual(len(row['critic_input']), 112)
        self.assertEqual(row['critic_input'][-4:], [1., -2., .25, 3.])
        self.assertEqual(row['contract'], 'mea_online_transport_v1')
        s['normalized_values'][0] = 88
        self.assertEqual(row['critic_input'][-4], 1.)

    def test_snapshot_digests_context_and_sources_are_checked_before_projection(self):
        m = self.runtime(); obs = observation()
        cases = [dict(source='live'), dict(context_id='f'*64), dict(binding_sha256='f'*64),
                 dict(normalization_id='other'), dict(coverage_basis='peak_frames'),
                 dict(end_frame=143399), dict(normalized_values=[1, 2, 3, 6]),
                 dict(feature_values=[-1, 0, 0, 0]), dict(feature_values=[1, 1, 2, 0]),
                 dict(reward=999)]
        for changes in cases:
            bad = snapshot(obs); bad.update(changes)
            bad['snapshot_id'] = digest({k:v for k,v in bad.items() if k != 'snapshot_id'})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.project_observation(obs, bad, config())
        bad = snapshot(obs); bad['feature_values'][0] += 1
        with self.assertRaises(ValueError):m.project_observation(obs, bad, config())

    def test_gae_keeps_terminal_boundaries_and_inputs_immutable(self):
        m = self.runtime()
        rows = [dict(value=.3, reward=1., duration=2, terminal=False, boundary=False),
                dict(value=.2, reward=-2., duration=3, terminal=True, boundary=True),
                dict(value=.7, reward=4., duration=1, terminal=False, boundary=False)]
        before=copy.deepcopy(rows)
        result=m.trajectory(rows, 5., .9, .95)
        self.assertEqual(rows, before)
        self.assertAlmostEqual(result[1]['advantage'], -2.2)
        self.assertAlmostEqual(result[0]['advantage'], 1+.9**2*.2-.3+(.9*.95)**2*(-2.2))
        self.assertAlmostEqual(result[2]['return'], 8.5)
        self.assertTrue(result[-1]['boundary'])
        with self.assertRaises(ValueError):m.trajectory(rows[:2], 7., .9, .95)

    def test_r2_and_stimulation_rejected_without_loading_torch(self):
        m = self.runtime()
        for key in ('r2_enabled', 'stimulation'):
            c=config();c[key]=True
            with self.subTest(key=key), self.assertRaises(ValueError):m.validate_runtime_config(c)


try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    class TinyPolicy:
        """Real differentiable CPU policy; no Qwen or hardware dependencies."""
        device_name='cpu'

        def __init__(self):
            torch.manual_seed(331)
            self.model=torch.nn.Linear(112, 2)
            self.model.bias.requires_grad_(False)
            self.reference_model=copy.deepcopy(self.model).requires_grad_(False)
            self.critic=torch.nn.Sequential(torch.nn.Linear(108,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
            from mea_online.m9_runtime import expand_online_critic
            expand_online_critic(self.critic)

        def distributions(self, pp, verbs, reference=False):
            x=torch.tensor([r['critic_input'] for r in pp], dtype=torch.float32)
            model=self.reference_model if reference else self.model
            vd=torch.distributions.Categorical(logits=model(x))
            td=torch.distributions.Categorical(logits=torch.zeros(len(pp),1))
            return vd,td

        def values(self, pp):
            from mea_online.m9_runtime import critic_values
            return critic_values(self,pp)

        def assert_reference_frozen(self):
            if any(p.requires_grad or p.grad is not None for p in self.reference_model.parameters()):
                raise RuntimeError('reference changed')

        def metadata(self):return dict(contract='mea_online_transport_v1', source='simulation')


def tiny_rows(policy):
    rows=[]
    with torch.no_grad():
        for i in range(8):
            p=dict(contract='mea_online_transport_v1', critic_input=[.01*(i+1)]*108+[1.,-2.,.25,3.],
                   branches=[[dict(verb='wait')],[dict(verb='rest')]], target_texts=[['wait'],['rest']],
                   input_ids=[4,5], messages=[dict(role='user',content='simulation')])
            p['prepared_sha256']=digest(p)
            verb=i%2;vd,td=policy.distributions([p],[verb]);rv,rt=policy.distributions([p],[verb],True)
            rows.append(dict(prepared=p,action=p['branches'][verb][0],verb=verb,target=0,
                value=float(policy.values([p])[0]),logp=float(vd.log_prob(torch.tensor([verb]))),
                verb_logp=float(vd.log_prob(torch.tensor([verb]))),reference_verb=rv.probs[0].tolist(),
                reference_target=rt.probs[0].tolist(),reward=(1. if verb==1 else -.3),duration=1,
                terminal=i==7,boundary=i==7))
    return rows


@unittest.skipUnless(torch is not None, 'CPU torch dependencies required')
class TorchRuntimeTests(unittest.TestCase):
    def core(self):
        from mea_online import m9_runtime as m
        self.assertTrue(hasattr(m, '_ModelCore'), 'model update/checkpoint core absent')
        c=config();c['training']=dict(epochs=2,critic_epochs=2,actor_lr=.001,critic_lr=.001)
        return m._ModelCore(TinyPolicy(), c)

    def test_extra_columns_start_at_zero_but_receive_training_gradients(self):
        from mea_online import m9_runtime as m
        self.assertTrue(hasattr(m,'expand_online_critic'), 'critic expansion absent')
        torch.manual_seed(7)
        critic=torch.nn.Sequential(torch.nn.Linear(108,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
        facts=torch.randn(1,108);expected=critic(facts).detach();old=critic[0].weight.detach().clone()
        m.expand_online_critic(critic)
        actual=critic(torch.cat([facts,torch.tensor([[1.,2.,3.,4.]])],1))
        # 108→112 changes FP32 reduction shape; use the registered 1e-6 limit.
        self.assertTrue(torch.allclose(expected,actual,rtol=0,atol=1e-6))
        self.assertTrue(torch.equal(old,critic[0].weight[:,:108]))
        self.assertEqual(int(torch.count_nonzero(critic[0].weight[:,108:])),0)
        actual.sum().backward()
        self.assertGreater(int(torch.count_nonzero(critic[0].weight.grad[:,108:])),0)

    def test_update_changes_actor_critic_keeps_base_reference_and_checkpoint_optimizers(self):
        core=self.core();rows=tiny_rows(core.policy);frozen=copy.deepcopy(rows);before=core.state()
        metrics=core.update(rows,0.,1)
        self.assertGreater(metrics['optimizer_steps'],0)
        self.assertGreater(metrics['critic_optimizer_steps'],0)
        self.assertEqual(rows,frozen)
        after=core.state()
        self.assertNotEqual(before['actor'],after['actor']);self.assertNotEqual(before['critic'],after['critic'])
        self.assertEqual(before['base'],after['base']);self.assertEqual(before['reference'],after['reference'])
        with tempfile.TemporaryDirectory() as tmp:
            result=core.checkpoint(Path(tmp)/'checkpoint','c'*64,1)
            self.assertTrue(result['roundtrip_verified']);self.assertTrue(result['optimizer_roundtrip_verified'])
            restored=self.core();restored.load_checkpoint(Path(tmp)/'checkpoint','c'*64)
            self.assertEqual(restored.state(),after)
            self.assertEqual(restored.optimizer_identity(),core.optimizer_identity())
            second=tiny_rows(restored.policy);stats=restored.update(second,0.,2)
            self.assertGreater(stats['optimizer_steps'],0)
            self.assertGreater(min(restored.optimizer_identity()['steps']['actor']),min(core.optimizer_identity()['steps']['actor']))
            with self.assertRaises(FileExistsError):core.checkpoint(Path(tmp)/'checkpoint','c'*64,1)

    def test_corrupt_old_statistics_and_frozen_input_stop_before_weights_change(self):
        for key in ('value','logp','verb_logp','reference_verb','prepared'):
            core=self.core();rows=tiny_rows(core.policy);before=core.state()
            if key=='reference_verb':rows[0][key]=[.9,.1]
            elif key=='prepared':rows[0][key]['critic_input'][0] += 1
            else:rows[0][key] += .2
            with self.subTest(key=key),self.assertRaises((ValueError,RuntimeError)):
                core.update(rows,0.,1)
            self.assertEqual(before,core.state())

    def test_frozen_parameter_damage_is_rejected_before_another_optimizer_step(self):
        core=self.core()
        with torch.no_grad():core.policy.model.bias.add_(.3)
        rows=tiny_rows(core.policy);before=core.state()
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            core.update(rows,0.,1)
        self.assertEqual(before['actor'],core.state()['actor'])
        self.assertEqual(core.optimizer_identity()['steps']['actor'],[])

    def test_checkpoint_tampering_is_rejected_before_loading_actor(self):
        core=self.core();core.update(tiny_rows(core.policy),0.,1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint';core.checkpoint(path,'c'*64,1)
            with (path/'actor.pt').open('ab') as stream:stream.write(b'changed')
            restored=self.core();before=restored.state()
            with self.assertRaisesRegex(ValueError,'checkpoint file changed'):
                restored.load_checkpoint(path,'c'*64)
            self.assertEqual(before,restored.state())

    def test_duplicate_checkpoint_json_keys_are_rejected_even_with_matching_file_hash(self):
        core=self.core();core.update(tiny_rows(core.policy),0.,1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint';core.checkpoint(path,'c'*64,1)
            p=path/'policy.json';p.write_text('{"update":999,'+p.read_text(encoding='utf-8')[1:],encoding='utf-8')
            manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            manifest['files']['policy.json']=hashlib.sha256(p.read_bytes()).hexdigest()
            (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                self.core().load_checkpoint(path,'c'*64)

    def test_native_r1_transition_matches_existing_reward_and_close_is_idempotent(self):
        from mea_online import m9_runtime as m
        self.assertTrue(hasattr(m,'M9Runtime'), 'runtime interface absent')
        from unittest.mock import patch
        from agent.m9_env import GodotEnv
        from agent.m9_contract import Reward
        root=Path(__file__).resolve().parents[3]
        godot=Path(os.environ.get('MEA_ONLINE_TEST_GODOT',str(root/'.tools/godot/Godot_v4.7.2-stable_win64_console.exe')))
        if not godot.exists():self.skipTest('native Godot unavailable')
        c=config();c.update(project=str(root),godot=str(godot))
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'_build_policy',return_value=TinyPolicy()):
            runtime=m.M9Runtime(c,tmp)
            try:
                with GodotEnv(str(godot),str(root),map_mode='infinite_freeze',navigation_steps=4) as native:
                    actual=runtime.reset(34171);expected=native.reset(34171)['observation']
                    self.assertEqual(actual,expected)
                    tracker=Reward(expected,'bottleneck',.99985)
                    for _ in range(3):
                        a={'verb':'wait'};result=runtime.step(a);nxt=native.step(a)
                        score=tracker.score(nxt['transition'],nxt['observation'])
                        self.assertEqual(result['transition'],nxt['transition'])
                        self.assertEqual(result['reward'],score['reward'])
                        self.assertEqual(result['duration'],score['elapsed_ticks'])
                        self.assertEqual(result['terminal'],nxt['terminated'])
            finally:runtime.close();runtime.close()

    def test_shards_count_each_decision_once_even_with_empty_ranks(self):
        from mea_online import distributed as d
        self.assertTrue(hasattr(d,'shard_minibatch'))
        for n in (1,7,8,16):
            ix=list(range(n));shards=[d.shard_minibatch(ix,rank,8) for rank in range(8)]
            self.assertEqual(sorted(i for shard in shards for i in shard),ix)
            # Weighted sum equals the single unique-minibatch objective.
            x=torch.tensor(1.3,requires_grad=True)
            full=sum((x*(i+1)-.1).square() for i in ix)/n
            full.backward();expected=float(x.grad);x.grad=None
            for shard in shards:
                if shard:(sum((x*(i+1)-.1).square() for i in shard)/n).backward()
            self.assertAlmostEqual(float(x.grad),expected,places=3)

    def test_shared_batch_two_replica_update_matches_single_objective_without_sockets(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from mea_online.m9_runtime import _ModelCore
        # Replace the external NCCL transport only, retaining real PPO/autograd,
        # real per-rank optimizers, partitioning, and weighted gradient reduction.
        class Group:
            def __init__(self):self.barrier=threading.Barrier(2,timeout=20);self.slots=[None,None]
            def exchange(self,rank,value,reduce):
                self.slots[rank]=value;self.barrier.wait()
                result=reduce(self.slots)
                self.barrier.wait()
                return result
        class InMemoryCollective:
            world=2
            def __init__(self,group,rank):self.group,self.rank=group,rank
            def require(self,condition,message):
                if not self.group.exchange(self.rank,condition,all):raise RuntimeError(message)
            def identical(self,value,label):
                if not self.group.exchange(self.rank,value,lambda vs:vs[0]==vs[1]):raise RuntimeError(label)
            def sum(self,value):
                return self.group.exchange(self.rank,torch.tensor(value,dtype=torch.float64),lambda vs:(vs[0]+vs[1]).tolist())
            def maximum(self,value):return self.group.exchange(self.rank,value,max)
            def gradients(self,parameters):
                for p in parameters:
                    grad=torch.zeros_like(p) if p.grad is None else p.grad
                    p.grad=self.group.exchange(self.rank,grad,lambda vs:vs[0]+vs[1]).clone()
        c=config();c['training']=dict(epochs=1,critic_epochs=1,actor_lr=.001,critic_lr=.001)
        single=_ModelCore(TinyPolicy(),c);replicas=[_ModelCore(TinyPolicy(),c) for _ in range(2)]
        rows=tiny_rows(single.policy);single.update(rows,0.,1)
        group=Group()
        for rank,core in enumerate(replicas):core.collective=InMemoryCollective(group,rank)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(core.update,copy.deepcopy(rows),0.,1) for core in replicas]
            results=[f.result(timeout=30) for f in futures]
        self.assertEqual(replicas[0].state(),replicas[1].state())
        self.assertEqual(replicas[0].optimizer_identity(),replicas[1].optimizer_identity())
        self.assertEqual(results[0]['unique_decisions'],8)
        for actual,want in zip(replicas[0].policy.model.parameters(),single.policy.model.parameters()):
            self.assertTrue(torch.allclose(actual,want,rtol=0,atol=1e-6))
        for actual,want in zip(replicas[0].policy.critic.parameters(),single.policy.critic.parameters()):
            self.assertTrue(torch.allclose(actual,want,rtol=0,atol=1e-6))

    def test_idle_replica_waits_for_parent_before_starting_collective_timeout(self):
        import threading
        import multiprocessing
        from unittest.mock import patch
        from mea_online import distributed as d
        self.assertTrue(hasattr(d,'receive_command'),'idle command wakeup absent')
        receive,send=multiprocessing.Pipe(duplex=False)
        called=threading.Event();result=[]
        def broadcast(values,src):called.set();values[0]={'op':'close'}
        with patch('torch.distributed.broadcast_object_list',side_effect=broadcast):
            thread=threading.Thread(target=lambda:result.append(d.receive_command(receive)))
            thread.start()
            self.assertFalse(called.wait(.05),'idle worker must not start NCCL timeout')
            send.send('broadcast');thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result,[{'op':'close'}])
        receive.close();send.close()


if __name__ == '__main__':unittest.main()
