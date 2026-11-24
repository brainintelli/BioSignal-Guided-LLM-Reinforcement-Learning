import importlib
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_mea_replay import fixture
from agent.mea_replay import AuditedRecording
from agent.mea_protocol import canonical,digest
from agent.mea_contract import observation_key
from tests.test_mea_p0 import GODOT,ROOT,observation


class OfflineTests(unittest.TestCase):
    def test_terminal_and_cutoff_gae_bootstrap(self):
        m=importlib.import_module('tools.mea_offline')
        rows=[dict(reward=1.,value=.3,duration=2,terminal=False,boundary=False),
              dict(reward=-2.,value=.2,duration=3,terminal=True,boundary=False)]
        m.finish_rows(rows,123.,.9,.95)
        self.assertEqual(rows[-1]['next_value'],0.)
        self.assertAlmostEqual(rows[-1]['advantage'],-2.2)
        self.assertAlmostEqual(rows[0]['advantage'],1+.9**2*.2-.3+(.9*.95)**2*(-2.2))
        cutoff=[dict(reward=1.,value=.3,duration=2,terminal=False,boundary=False)]
        m.finish_rows(cutoff,4.,.9,.95)
        self.assertAlmostEqual(cutoff[0]['return'],1+.9**2*4)

    @unittest.skipUnless(GODOT.exists(),'native Godot required')
    def test_native_recorded_roundtrip_and_inner_tampering(self):
        m=importlib.import_module('tools.mea_offline')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();expected=fixture(source)
            out=root/'run'
            report=m.collect(out,source,godot=GODOT,project=ROOT,steps=2,manifest_sha256=expected)
            self.assertEqual(report['snapshots'],3);self.assertFalse(report['model_training_performed'])
            result=m.replay(out,source,godot=GODOT,project=ROOT,manifest_sha256=expected)
            self.assertTrue(result['passed']);self.assertEqual(result['reward_exact_matches'],2)
            rows=[json.loads(s) for s in (out/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['next_snapshot'],rows[1]['snapshot'])
            self.assertTrue(all(r['reward_record']['total_reward']==r['r1_score']['reward'] for r in rows))
            rows[0]['prepared']['critic_input'][-1]+=1
            (out/'rollout.jsonl').write_text(''.join(canonical(x)+'\n' for x in rows),encoding='utf-8')
            manifest=json.loads((out/'manifest.json').read_text());manifest['files']['rollout.jsonl']=m.sha(out/'rollout.jsonl')
            (out/'manifest.json').write_text(canonical(manifest))
            with self.assertRaises(ValueError):m.replay(out,source,godot=GODOT,project=ROOT,manifest_sha256=expected)

    @unittest.skipUnless(GODOT.exists(),'native Godot required')
    def test_replay_pins_source_and_reconciles_report(self):
        m=importlib.import_module('tools.mea_offline')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();expected=fixture(source);out=root/'run'
            m.collect(out,source,godot=GODOT,project=ROOT,steps=2,manifest_sha256=expected)
            with self.assertRaises(ValueError):m.replay(out,source,godot=GODOT,project=ROOT)
            report=json.loads((out/'report.json').read_text());report.update(status='invalid',alpha=.2,
                stimulation=True,device_commands_sent=9,r2_enabled=True,elapsed_ticks=999,terminated=True)
            (out/'report.json').write_text(canonical(report),encoding='utf-8')
            manifest=json.loads((out/'manifest.json').read_text());manifest['files']['report.json']=m.sha(out/'report.json')
            (out/'manifest.json').write_text(canonical(manifest))
            with self.assertRaises(ValueError):m.replay(out,source,godot=GODOT,project=ROOT,manifest_sha256=expected)

    def test_every_ppo_probability_input_is_recomputed(self):
        m=importlib.import_module('tools.mea_offline')
        base=dict(logp=-1.,verb_logp=-.5,value=0.,reference_verb=[.2,.8],reference_target=[.4,.6])
        self.assertEqual(m.validate_policy_statistics([base],[dict(base)]),0.)
        for key,value in [('logp',2.),('verb_logp',3.),('value',4.),('reference_verb',[.5,.5]),
                          ('reference_target',[.9,.1]),('reference_verb',[.2]),('logp',float('nan'))]:
            changed=dict(base);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_policy_statistics([base],[changed])

    @unittest.skipUnless(GODOT.exists(),'native Godot required')
    def test_completion_claims_are_derived_from_evidence(self):
        m=importlib.import_module('tools.mea_offline')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();expected=fixture(source);out=root/'run'
            report=m.collect(out,source,godot=GODOT,project=ROOT,steps=2,manifest_sha256=expected)
            rows=[json.loads(s) for s in (out/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
            bootstrap=json.loads((out/'bootstrap.json').read_text(encoding='utf-8'))
            for key,value in [('status','invalid'),('alpha',.2),('stimulation',True),('r2_enabled',True),
                              ('device_commands_sent',9),('elapsed_ticks',999),('terminated',True),
                              ('snapshots',99),('action_source','model'),('model_training_performed',True)]:
                changed=dict(report);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_completion_report(changed,rows,bootstrap,2)

    def test_alpha_and_exhaustion_reject_before_outputs(self):
        m=importlib.import_module('tools.mea_offline')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();expected=fixture(source)
            for kwargs in [dict(alpha=.9,steps=2),dict(alpha=1.,steps=3)]:
                with self.assertRaises(ValueError):m.collect(root/'out',source,manifest_sha256=expected,**kwargs)
                self.assertFalse((root/'out').exists())


try:
    import torch
    HAS_TORCH=True
except ImportError:HAS_TORCH=False


@unittest.skipUnless(HAS_TORCH,'GPU server ML environment required')
class PolicyBridgeTests(unittest.TestCase):
    def test_critic_values_are_batch_invariant_without_losing_gradients(self):
        from types import SimpleNamespace
        from agent.mea_policy import RecordingPolicy
        from agent.mea_replay import RECORDING_CONTRACT
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(79)
        critic=torch.nn.Sequential(torch.nn.Linear(112,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1)).to(device)
        with torch.no_grad():critic[-1].weight.mul_(100)
        facts=torch.randn(17,112).tolist();prepared=[dict(contract=RECORDING_CONTRACT,critic_input=f) for f in facts]
        holder=SimpleNamespace(critic=critic,device_name=device)
        single=torch.cat([RecordingPolicy.values(holder,[r]) for r in prepared])
        batch=RecordingPolicy.values(holder,prepared)
        self.assertTrue(torch.equal(batch,single),float((batch-single).abs().max()))
        batch.sum().backward()
        self.assertGreater(int(torch.count_nonzero(critic[0].weight.grad[:,108:])),0)

    def test_zero_expansion_preserves_values_and_freezes_new_column_initialization(self):
        from agent.mea_policy import expand_recording_critic
        torch.manual_seed(5);critic=torch.nn.Sequential(torch.nn.Linear(108,128),torch.nn.Tanh(),torch.nn.Linear(128,1))
        before=critic(torch.randn(3,108));facts=torch.randn(3,108);expected=critic(facts).detach();old=critic[0].weight.detach().clone()
        expand_recording_critic(critic)
        actual=critic(torch.cat((facts,torch.randn(3,4)*1000),1))
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(critic[0].weight[:,:108],old))
        self.assertTrue(torch.equal(critic[0].weight[:,108:],torch.zeros(128,4)))

    def test_preparation_requires_one_snapshot_per_observation(self):
        from agent.mea_policy import prepare_recordings
        with self.assertRaises(ValueError):prepare_recordings(None,[observation()],[])
