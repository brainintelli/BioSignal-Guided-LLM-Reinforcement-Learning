import unittest
from tools.m8x_live_sampler_probe import select_pairs, require_accepted
from tools.check_m8x_live_link import compare_decision, require_restart, exercise, audit_log

class ProbeTests(unittest.TestCase):
    def test_live_gpu_work_blocks_sampler_without_loading_torch(self):
        from unittest.mock import patch
        from tools.m8x_live_sampler_probe import require_idle_gpu
        with patch('tools.m8x_live_sampler_probe.subprocess.check_output',return_value='1780066, training, 45159 MiB'):
            with self.assertRaisesRegex(RuntimeError,'GPU compute work left untouched'):require_idle_gpu()
        with patch('tools.m8x_live_sampler_probe.subprocess.check_output',return_value=''):
            require_idle_gpu()
    def test_gate_rejects_selection_and_forged_accept(self):
        for data in ({}, {'phase':'selection','decision':'accept','release_authorized':True,'evidence_complete':True}):
            with self.assertRaises(ValueError): require_accepted(data)
    def test_training_pairs_and_negative_sources(self):
        c=dict(train_seed_start=10,train_seed_count=4,validation_seeds=[20],test_seeds=[30])
        rows=[dict(update=1,env=e,seed=10+e,observation={'tick':t},transition={'after':{'tick':t+1}},terminal=False,truncated=False) for t in [0,1] for e in range(4)]
        self.assertEqual(len(select_pairs(rows,c)),4)
        for field,value in [('seed',20),('terminal',True),('truncated',True)]:
            import copy
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):select_pairs(bad,c)
        bad=[dict(r) for r in rows];bad[4]=dict(bad[4],seed=11)
        with self.assertRaises(ValueError):select_pairs(bad,c)
        with self.assertRaises(ValueError):select_pairs(rows[:-1],c)
    def test_exact_compare_and_actual_restart(self):
        expected={'action':{'verb':'wait'},'rng_after_sha256':'a','verb_probabilities':[.5,.5]}
        compare_decision(expected,dict(expected,latency_ms=7))
        with self.assertRaises(ValueError):compare_decision(expected,dict(expected,rng_after_sha256='b'))
        old={'pid':10,'start_ticks':'1'}
        with self.assertRaises(ValueError):require_restart(old,old)
        require_restart(old,{'pid':11,'start_ticks':'2'})

    def test_http_sequence_rejects_changed_cache(self):
        from unittest.mock import patch
        decision={'source':'model','raw_valid':True,'rng_after_sha256':'a'}
        replies=[(200,decision),(200,decision),(400,{'detail':'Changed observation at an already decided tick'}),(200,decision)]
        obs={'inventory':{'berries':0}}
        with patch('tools.check_m8x_live_link.request',side_effect=replies) as req:
            self.assertEqual(exercise('unused',obs,{}, {'rng_after_sha256':'a'})['decision'],decision)
            self.assertEqual(req.call_args_list[2].args[2]['inventory']['berries'],1)
            self.assertEqual(obs['inventory']['berries'],0)
        for n in [1,3]:
            changed=list(replies);changed[n]=(200,dict(decision,rng_after_sha256='b'))
            with patch('tools.check_m8x_live_link.request',side_effect=changed):
                with self.assertRaises(ValueError):exercise('unused',obs,{}, {'rng_after_sha256':'a'})

    def test_missing_log_is_allowed_only_before_first_request(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as folder:
            path=Path(folder)/'missing.jsonl'
            self.assertEqual(audit_log(path,[{'session_id':'m8x-probe-a'}],[[]],'actor','release'),{'m8x-probe-a':0})
            with self.assertRaises(ValueError):audit_log(path,[{'session_id':'m8x-probe-a'}],[[{}]],'actor','release')

    def test_other_protocols_cannot_authorize_exploration_probe(self):
        for protocol in ('m8_balanced_reliable_v1', 'm8b_sil_reliable_v1', 'unknown'):
            report=dict(protocol=protocol,phase='test',decision='accept',release_authorized=True,evidence_complete=True)
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):require_accepted(report)

    def test_preflight_calls_complete_release_verifier(self, protocol='m8_exploration_reliable_v1'):
        from unittest.mock import patch
        from tools.m8x_live_sampler_probe import preflight
        from pathlib import Path
        import tempfile,json
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests') as folder:
            root=Path(folder);cp=root/'checkpoint_1';cp.mkdir();(cp/'policy.json').write_text('{}')
            report=dict(protocol=protocol,phase='test',decision='accept',release_authorized=True,evidence_complete=True,
                config={'output':folder,'model':'model'},selected_checkpoint='checkpoint_1',selected_actor_sha256='actor',execution_identity={})
            path=root/'result.json';path.write_text(json.dumps(report))
            # A forged true bit cannot bypass the full verifier; no GPU is loaded.
            with patch('tools.m8x_live_sampler_probe.verify_release',side_effect=ValueError('Complete result does not reproduce')) as gate:
                with self.assertRaisesRegex(ValueError,'does not reproduce'):preflight(path,folder)
                gate.assert_called_once()

if __name__=='__main__': unittest.main()
