"""Resume imports are immutable, complete, and cannot change the study gate."""
import copy
import gzip
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from agent.m8s3_training import sha, write
from agent.m9_metrics import summarize
from tools.m9_campaign import digest, task_spec


class ResumeTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('tools.m9_resume')
        except ModuleNotFoundError:
            self.fail('Verified campaign resume is missing')

    def fixture(self, directory):
        root = Path(directory)
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/m9_research_attempt1.json').read_text(encoding='utf-8'))
        config.update(project=str(root/'old-project'), output=str(root/'old-output'), job=str(root/'old-job'))
        contract = root/'old-project/agent/m9_contract.py'
        contract.parent.mkdir(parents=True)
        contract.write_text('unchanged contract')
        reg = dict(config=config, source_hashes={'agent/m9_contract.py': sha(contract)}, input_files={},
                   test_seeds=config['test_seeds'], warm_actor_sha256='warm')
        reg['registration_sha256'] = digest(reg)
        rp = root/'previous.json';write(rp, reg)
        (root/'old-job').mkdir()
        write(root/'old-job/finished.json', dict(status='failed', seconds=100,
              error='RuntimeError: explore_state_compact_r1 exited 1',
              stages=[dict(stage='explore_state_compact_r1', exit_code=1)]))
        for arm, rep, complete in [('control', 0, True), ('state_compact', 1, False)]:
            spec = task_spec(reg, 'explore', arm, rep)
            p = Path(spec['output']);p.mkdir(parents=True)
            write(p/'config.json', spec)
            metrics = [dict(update=i, decisions=i*2048, ticks=i*2048) for i in range(1, 9)]
            (p/'metrics.jsonl').write_text('\n'.join(json.dumps(m) for m in metrics)+'\n')
            checkpoint = p/'checkpoint_0008';checkpoint.mkdir()
            (checkpoint/'weights').write_bytes(b'weights-'+arm.encode())
            write(checkpoint/'manifest.json', dict(files={'weights':sha(checkpoint/'weights')},
                  actor_sha256='trained-'+arm, update=8, registration_sha256=reg['registration_sha256']))
            write(checkpoint/'reload_probe.json', dict(rows=[{}], logp=[-1.]))
            episodes = [dict(seed=s, replicate=0, survival_ticks=5400, terminal=False,
                        alive_at_horizon=True, alive_at_1800=True) for s in spec['evaluation_seeds']]
            for kind in ['initial', 'evaluation'] if complete else ['initial']:
                trace = p/(kind+'.jsonl.gz')
                with gzip.open(trace, 'wt') as f:f.write('{}\n')
                write(p/(kind+'.json'), dict(episodes=episodes, summary=summarize(episodes),
                    horizon=5400, trace_sha256=sha(trace), registration_sha256=reg['registration_sha256'],
                    actor_sha256='warm' if kind=='initial' else 'trained-'+arm))
            if complete:
                write(p/'training_complete.json', dict(updates=8, decisions=16384,
                      checkpoint=str(checkpoint), final_actor='trained-'+arm))
                write(checkpoint/'independent_reload.json', dict(passed=True,logp_error=0.,
                      actor_sha256='trained-'+arm, registration_sha256=reg['registration_sha256']))
            else:
                (p/'evaluation.jsonl.gz').write_bytes(b'incomplete retained evidence')
        new = copy.deepcopy(config)
        new.update(project=str(root/'new-project'), output=str(root/'new-output'), job=str(root/'new-job'),
                   resume_from=str(rp), hard_wall_seconds=config['hard_wall_seconds']-100)
        new['common']['max_tokens'] = 4096
        return new, reg

    def test_complete_is_imported_and_failed_evaluation_retains_checkpoint(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            c, old = self.fixture(directory)
            resume, inputs = module.collect_resume(c)
            self.assertEqual(resume['runs']['control_r0']['kind'], 'complete')
            self.assertEqual(resume['runs']['state_compact_r1']['kind'], 'evaluate_saved')
            self.assertEqual(resume['reused_updates'], 16)
            self.assertEqual(resume['reused_decisions'], 32768)
            reg = dict(config=c, resume=resume, registration_sha256='new')
            spec = task_spec(reg, 'explore', 'control', 0)
            before = {k: sha(v['path']) for k,v in inputs.items()}
            module.import_completed(spec)
            self.assertEqual(json.loads((Path(spec['output'])/'evaluation.json').read_text())['registration_sha256'],old['registration_sha256'])
            self.assertTrue((Path(spec['output'])/'reuse.json').exists())
            self.assertEqual(before, {k: sha(v['path']) for k,v in inputs.items()})
            with self.assertRaises(FileExistsError):module.import_completed(spec)
            self.assertEqual(task_spec(reg,'explore','state_compact',1)['resume_kind'],'evaluate_saved')
            self.assertNotIn('resume_kind',task_spec(reg,'explore','action_macro4',1))
            self.assertNotIn('resume_kind',task_spec(reg,'confirm','control',2))

    def test_changed_rules_or_gate_or_used_final_tests_are_rejected(self):
        module = self.module()
        for field in ('arms', 'phases', 'final_gate', 'test_claim'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                c, old = self.fixture(directory)
                if field=='arms':c['arms']['control']['reward_variant']='survival'
                if field=='phases':c['phases']['explore']['updates']=9
                if field=='final_gate':c['final_gate']='weaker gate'
                if field=='test_claim':write(Path(old['config']['output'])/'test_claim.json',{})
                with self.assertRaises(ValueError):module.collect_resume(c)

    def test_recovery_keeps_training_and_never_copies_partial_assessment(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            c,old=self.fixture(directory)
            resume,inputs=module.collect_resume(c)
            reg=dict(config=c,resume=resume,registration_sha256='new')
            spec=task_spec(reg,'explore','state_compact',1)
            source=Path(spec['resume_source']);out=Path(spec['output'])
            partial=(source/'evaluation.jsonl.gz').read_bytes()
            module.copy_evaluation_recovery(spec)
            self.assertEqual((source/'evaluation.jsonl.gz').read_bytes(),partial)
            self.assertFalse((out/'evaluation.jsonl.gz').exists())
            self.assertFalse((out/'evaluation.json').exists())
            self.assertEqual(sha(out/'checkpoint_0008/weights'),sha(source/'checkpoint_0008/weights'))
            self.assertEqual(sha(out/'metrics.jsonl'),sha(source/'metrics.jsonl'))
            with self.assertRaises(FileExistsError):module.copy_evaluation_recovery(spec)

    def test_file_changed_after_registration_cannot_be_imported(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as directory:
            c,old=self.fixture(directory)
            resume,inputs=module.collect_resume(c)
            spec=task_spec(dict(config=c,resume=resume,registration_sha256='new'),'explore','control',0)
            (Path(spec['resume_source'])/'evaluation.json').write_text('{}')
            with self.assertRaises(ValueError):module.import_completed(spec)
            self.assertFalse(Path(spec['output']).exists())

    def test_incomplete_optimizer_progress_or_changed_checkpoint_rejected(self):
        module = self.module()
        for broken in ('metrics', 'weights', 'reload'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as directory:
                c, old = self.fixture(directory)
                p=Path(old['config']['output'])/'explore/control_r0'
                if broken=='metrics':(p/'metrics.jsonl').write_text('{}\n')
                if broken=='weights':(p/'checkpoint_0008/weights').write_bytes(b'changed')
                if broken=='reload':write(p/'checkpoint_0008/independent_reload.json',dict(passed=False))
                with self.assertRaises(ValueError):module.collect_resume(c)


if __name__=='__main__':unittest.main()
