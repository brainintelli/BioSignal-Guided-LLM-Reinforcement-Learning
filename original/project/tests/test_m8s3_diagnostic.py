"""Diagnostic guards: equal work, no stale initialization, real fixed padding."""
import copy
import importlib
import unittest


class SurvivalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.d = importlib.import_module('agent.m8s3_diag_contract')

    def test_padding_repeats_only_physical_rows_and_never_truncates(self):
        inputs = [[1,2,3],[4],[5,6],[7],[8,9]]
        batches = list(self.d.physical_batches(inputs, max_tokens=5, pad_id=0))
        self.assertEqual([b['real_rows'] for b in batches], [4,1])
        self.assertEqual(batches[0]['input_ids'][0], [0,0,1,2,3])
        self.assertEqual(batches[0]['attention_mask'][1], [0,0,0,0,1])
        self.assertEqual(batches[1]['input_ids'], [[0,0,0,8,9]]*4)
        self.assertEqual(inputs[0], [1,2,3])
        for value in ([], [[]], [[1]*6]):
            with self.assertRaises(ValueError):
                list(self.d.physical_batches(value, max_tokens=5, pad_id=0))

    def rows(self, seconds):
        return [dict(decisions=128,optimizer_steps=8,critic_optimizer_steps=8,
                     rollout_seconds=seconds/3,update_seconds=seconds*2/3,
                     finite=True,zero_update_logp_max=0.,pre_update_all_logp_max=0.,
                     refresh_error=0.,kl_early_stop=False,approx_kl_max=.001) for _ in range(3)]

    def test_speed_gate_requires_equal_actual_work_not_requested_budget(self):
        a,b=self.rows(100),self.rows(40)
        self.assertTrue(self.d.speed_gate(a,b)['passed'])
        b[0]['optimizer_steps']=7
        self.assertFalse(self.d.speed_gate(a,b)['passed'])
        self.assertFalse(self.d.speed_gate(a,b)['equal_work'])

    def test_speed_gate_rejects_numerical_failure_or_stopped_updates(self):
        for mutation in ({'refresh_error':.001},{'pre_update_all_logp_max':.01},
                         {'finite':False},{'kl_early_stop':True},{'critic_optimizer_steps':7}):
            with self.subTest(mutation=mutation):
                a,b=self.rows(100),self.rows(40);b[1].update(mutation)
                self.assertFalse(self.d.speed_gate(a,b)['passed'])
        self.assertFalse(self.d.speed_gate(self.rows(100),self.rows(50))['passed'])

    def test_config_rejects_old_world_resume_and_hidden_budget_reduction(self):
        c=self.d.fixed_config('/remote/project','/remote')
        self.assertEqual(c['model'],'/remote/models/Qwen3.5-0.8B')
        self.d.validate_config(c)
        self.d.validate_config(self.d.fixed_config('/remote/project','/remote','native'))
        for mutation in ({'profile':'open3'},{'world_version':5},{'reset_critic':False},
                         {'fresh_optimizers':False},{'updates':2},{'epochs':1},
                         {'resume':'old_optimizer'},{'prefix_layout':'system_scene'},
                         {'formal_training_status':'running'},{'actor_lr':1e-4}):
            bad=copy.deepcopy(c);bad.update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.d.validate_config(bad)

    def test_registration_digest_covers_all_content(self):
        r={'config':{'updates':3},'source_hashes':{'a.py':'b'*64}}
        r['registration_sha256']=self.d.registration_digest(r)
        self.d.verify_registration_digest(r)
        r['config']['updates']=2
        with self.assertRaises(ValueError):self.d.verify_registration_digest(r)


if __name__=='__main__':unittest.main()
