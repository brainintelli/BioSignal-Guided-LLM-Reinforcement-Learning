"""Synthetic release-boundary tests; these fixtures are not model performance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import m8x_release
from agent.m8x_contract import balance_identity, REWARD_CONFIG


class ExplorationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'
        self.run = self.root / 'run'
        self.checkpoint = self.run / 'checkpoint_0048'
        self.checkpoint.mkdir(parents=True)
        rules = Path(__file__).resolve().parents[1] / 'configs/m8_exploration_v1.json'
        (self.root / 'configs').mkdir()
        (self.root / 'configs/m8_exploration_v1.json').write_bytes(rules.read_bytes())
        self.world = balance_identity(self.root)
        self.actor = 'a' * 64
        self.meta = dict(contract='open_world_exploration_v1', actor_sha256=self.actor,
            reference_actor_sha256=m8x_release.PARENT_ACTOR_SHA256,
            inference_layout='fixed_batch4_left_pad_v1', physical_forward_shape=[4,1536],
            reward_config=copy.deepcopy(REWARD_CONFIG),kernel_identity={"fixture":True},max_tokens=1536, task_goals=False, policy_override=False, **self.world)
        self.execution = dict(base_weight_sha256=m8x_release.BASE_WEIGHT_SHA,
            runtime={'fixture':'CPU','kernel_identity':{'fixture':True}},tokenizer_files_sha256={'fixture':'tokenizer'})
        self.report = dict(protocol='m8_exploration_reliable_v1',phase='test',decision='accept',
            release_authorized=True,evidence_complete=True,selected_checkpoint=self.checkpoint.name,
            selected_actor_sha256=self.actor,parent_actor_sha256=m8x_release.PARENT_ACTOR_SHA256,
            execution_identity=self.execution,**{k:v for k,v in self.world.items() if k!='profile'},
            config=dict(project=str(self.root),model=str(self.model),output=str(self.run),
                contract='open_world_exploration_v1',max_tokens=1536,temperature=1,
                **{k:v for k,v in self.world.items() if k!='balance_sha256'}),
            publication_request={'phase':'test','training_run':str(self.run)})
        self.path = self.root / 'verification.json'

    def verify(self, report=None, meta=None, **changes):
        report = self.report if report is None else report
        meta = self.meta if meta is None else meta
        self.path.write_text(json.dumps(report),encoding='utf-8')
        (self.checkpoint/'policy.json').write_text(json.dumps(meta),encoding='utf-8')
        args=dict(verification=self.path,project=self.root,checkpoint=self.checkpoint,
            model=self.model,actor_sha256=self.actor,policy_metadata=meta,execution_identity=self.execution)
        args.update(changes)
        return m8x_release.verify_release(**args)

    def test_accepted_evidence_is_rebuilt_before_release(self):
        with patch.object(m8x_release,'build_result',return_value=self.report) as rebuild:
            self.assertEqual(self.verify(),self.report)
        rebuild.assert_called_once_with(self.report['publication_request'],self.root)

    def test_reward_and_runtime_metadata_cannot_drift(self):
        for key in ('reward_config','kernel_identity'):
            with self.subTest(key=key),patch.object(m8x_release,'build_result',return_value=self.report):
                with self.assertRaises(ValueError):self.verify(meta={**self.meta,key:{}})

    def test_unknown_protocol_never_reaches_any_comparator(self):
        with patch.object(m8x_release, 'build_result') as old:
            with self.assertRaisesRegex(ValueError, 'Unsupported release protocol'):
                self.verify({**self.report, 'protocol': 'unknown'})
            old.assert_not_called()

    def test_unverified_gate_or_different_report_cannot_release(self):
        for key,value in [('phase','selection'),('decision','retain_parent'),('release_authorized',False),
                          ('evidence_complete',False),('protocol','m8_reliable_v1')]:
            changed={**self.report,key:value}
            with self.subTest(key=key),patch.object(m8x_release,'build_result',return_value=changed):
                with self.assertRaises(ValueError):self.verify(changed)
        with patch.object(m8x_release,'build_result',return_value=self.report):
            with self.assertRaises(ValueError):self.verify({**self.report,'unverified':'extra'})

    def test_actual_world_identity_bound_in_report_config_and_policy(self):
        for key,value in [('profile','open1'),('world_version',2),('environment_version','legacy'),
                          ('balance_sha256','b'*64)]:
            for location in ('report','config','policy'):
                if (key=='profile' and location=='report') or (key=='balance_sha256' and location=='config'):
                    continue
                report=copy.deepcopy(self.report);meta=copy.deepcopy(self.meta)
                (meta if location=='policy' else report['config'] if location=='config' else report)[key]=value
                with self.subTest(key=key,location=location),patch.object(m8x_release,'build_result',return_value=report):
                    with self.assertRaises(ValueError):self.verify(report,meta)

    def test_identity_shape_paths_and_rules_tampering_fail(self):
        for changes in [dict(actor_sha256='c'*64),dict(checkpoint=self.run/'checkpoint_0008'),
                        dict(model=self.root/'other'),dict(execution_identity={}),
                        dict(meta={**self.meta,'physical_forward_shape':[1,1536]}),
                        dict(meta={**self.meta,'contract':'open_world_semantic_v4'})]:
            with self.subTest(changes=changes),patch.object(m8x_release,'build_result',return_value=self.report):
                with self.assertRaises(ValueError):self.verify(**changes)
        rules=self.root/'configs/m8_exploration_v1.json'
        rules.write_bytes(rules.read_bytes()+b'\n')
        with patch.object(m8x_release,'build_result',return_value=self.report):
            with self.assertRaises(ValueError):self.verify()


if __name__=='__main__':unittest.main()
