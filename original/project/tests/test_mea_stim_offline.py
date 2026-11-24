import unittest
from agent.mea_stim_offline import assignment,cell_config,validate_budget,analyze,features,validate_window,PROTOCOL

class StimOfflineTests(unittest.TestCase):
    def test_final_analysis_failure_or_incomplete_never_marks_batch_passed(self):
        import json,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from tools.mea_stim_campaign import batch
        for result in [ValueError('receipt mismatch'),{'status':'incomplete','positive_evidence':False}]:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);args=SimpleNamespace(root=root,source=root,pin='pin',model_config=root/'m',project=root,godot=root/'godot')
                with patch('tools.mea_stim_campaign.registration'),patch('tools.mea_stim_campaign.subprocess.run',return_value=SimpleNamespace(returncode=0)),patch('tools.mea_stim_campaign.analysis',side_effect=result if isinstance(result,Exception) else None,return_value=result):
                    try:batch(args)
                    except ValueError:pass
                status=json.loads((root/'batch_status.json').read_text())
                self.assertFalse(status['passed'])
                self.assertTrue(status['complete'])
                self.assertIn('error',status)

    def test_short_trainer_requires_original_tensors_and_finite_budget(self):
        from tools import mea_stim_train as t
        from agent.mea_long_ablation import ORIGINAL_TENSORS
        t.validate_common_initial(ORIGINAL_TENSORS)
        with self.assertRaises(ValueError):t.validate_common_initial(ORIGINAL_TENSORS|{'actor':'replacement'})
        self.assertEqual(t.PROTOCOL['rounds'],3)
        self.assertEqual(t.window_split(600)['train_current'],[0,384])
        with self.assertRaises(ValueError):t.window_split(599)

    def test_cross_evaluation_schedule_matches_between_training_conditions(self):
        from tools.mea_stim_campaign import evaluation_schedule
        for context in ['before','after']:
            a=evaluation_schedule(0,'before',context);b=evaluation_schedule(0,'after',context)
            self.assertEqual(a,b)
            self.assertEqual(a['order'],assignment('eval'))
        with self.assertRaises(ValueError):evaluation_schedule(0,'after','train')

    def test_prompt_does_not_falsely_label_post_stim_data_unstimulated(self):
        from dataclasses import replace
        from agent.mea_ablation import encode_arm
        from agent.mea_replay import RecordingSnapshot
        from agent.mea_snapshot import Normalization
        from agent.mea_contract import observation_key
        from tests.test_mea_p0 import observation
        obs=observation()
        before=RecordingSnapshot('a','stim28080_before',observation_key(obs),'m',0,0,804,20000,0,20000,'s',(0.,0.,0.,0.),Normalization.identity())
        after=replace(before,session_id='stim28080_after')
        a=encode_arm(obs,before,'real');b=encode_arm(obs,after,'real')
        self.assertEqual(a['messages'],b['messages'])
        self.assertNotIn('无刺激',b['messages'][1]['content'])
        self.assertIn('与本次游戏动作无因果对应',b['messages'][1]['content'])

    def test_finite_separated_splits_and_bootstrap(self):
        train=assignment('train');evaluation=assignment('eval')
        self.assertEqual(train,list(range(385)))
        self.assertEqual(evaluation,list(range(445,596)))
        self.assertFalse(set(train)&set(evaluation))
        self.assertEqual(min(evaluation)-max(train)-1,60)
        with self.assertRaises(ValueError):assignment('unknown')

    def test_paired_seeds_independent_of_training_arm(self):
        configs=[cell_config(2,a) for a in PROTOCOL['arms']]
        for field in ['initial_seed','world_seed','action_seed','ppo_seed','eval_world_seed','eval_action_seed']:
            self.assertEqual(len({c[field] for c in configs}),1)
        self.assertEqual({c['policy_arm'] for c in configs},{'real','none'})
        for block,arm in [(8,'before'),(0,'real'),(-1,'none')]:
            with self.assertRaises(ValueError):cell_config(block,arm)

    def test_budget_checks_actual_updates_and_round_count(self):
        validate_budget(384,[{'actor':2,'critic':16}]*3)
        for count,steps in [(1024,[{'actor':2,'critic':16}]*3),(384,[{'actor':1,'critic':16}]*3),(384,[{'actor':2,'critic':16}]*2)]:
            with self.assertRaises(ValueError):validate_budget(count,steps)

    def test_native_amplitude_conversion_and_window_bounds(self):
        values=features([[100,1,-3.],[102,2,5.]],channels=2,frames=20000,rate=20000,lsb_volts=2e-6)
        self.assertEqual(values,[2.,1.,1.,8.])
        valid={'start':100,'end':20100,'events':[[100,1,-6.],[102,2,10.]],'features':values}
        validate_window(valid,0,100,[(1,10),(2,20)])
        for change in [{'events':[[99,1,-6.]]},{'events':[[100,640,-6.]]},{'end':20101},{'features':[2.,1.,1.,9.]}]:
            with self.assertRaises(ValueError):validate_window(valid|change,0,100,[(1,10),(2,20)])

    def test_missing_cells_never_positive(self):
        self.assertEqual(analyze([])['status'],'incomplete')
        self.assertFalse(analyze([])['positive_evidence'])
        cells=[dict(block=b,arm=a,pre={'before':1.,'after':1.},post={'before':2.,'after':2.}) for b in range(8) for a in PROTOCOL['arms']]
        result=analyze(cells)
        self.assertFalse(result['positive_evidence'])
        self.assertEqual(result['contrasts']['after_minus_before']['post']['mean_difference'],0.)
        with self.assertRaises(ValueError):analyze(cells+[cells[0]])

if __name__=='__main__':unittest.main()
