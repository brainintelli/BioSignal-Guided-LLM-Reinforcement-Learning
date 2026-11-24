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
            result=m.replay(out,source,godot=GODOT,project=ROOT)
            self.assertTrue(result['passed']);self.assertEqual(result['reward_exact_matches'],2)
            rows=[json.loads(s) for s in (out/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['next_snapshot'],rows[1]['snapshot'])
            self.assertTrue(all(r['reward_record']['total_reward']==r['r1_score']['reward'] for r in rows))
            rows[0]['prepared']['critic_input'][-1]+=1
            (out/'rollout.jsonl').write_text(''.join(canonical(x)+'\n' for x in rows),encoding='utf-8')
            manifest=json.loads((out/'manifest.json').read_text());manifest['files']['rollout.jsonl']=m.sha(out/'rollout.jsonl')
            (out/'manifest.json').write_text(canonical(manifest))
            with self.assertRaises(ValueError):m.replay(out,source,godot=GODOT,project=ROOT)

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
