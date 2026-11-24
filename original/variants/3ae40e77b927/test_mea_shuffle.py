import importlib, unittest
from pathlib import Path

class ShuffleTests(unittest.TestCase):
    def module(self):
        spec=importlib.util.find_spec('agent.mea_shuffle')
        self.assertIsNotNone(spec,'Missing follow-up protocol')
        return importlib.import_module('agent.mea_shuffle')

    def test_complete_window_derangement_does_not_leak_or_shuffle_bootstrap(self):
        m=self.module();orders=[]
        for b in range(8):
            order=m.train_order(b,'shuffled_after');orders.append(tuple(order))
            self.assertEqual(sorted(order[:384]),list(range(384)))
            self.assertTrue(all(i!=v for i,v in enumerate(order[:384])))
            self.assertEqual(order[384],384)
            self.assertEqual(order,m.train_order(b,'shuffled_after'))
            self.assertEqual(m.train_order(b,'after'),list(range(385)))
        self.assertEqual(len(set(orders)),8)

    def test_paired_new_seeds_and_invalid_registration(self):
        m=self.module()
        for b in range(8):
            cfgs=[m.cell_config(b,a) for a in m.PROTOCOL['arms']]
            for key in ['initial_seed','world_seed','action_seed','ppo_seed','eval_world_seed','eval_action_seed']:
                self.assertEqual(len({c[key] for c in cfgs}),1)
            self.assertEqual(cfgs[0]['world_seed'],8100100+1000*b)
            self.assertEqual(cfgs[0]['eval_world_seed'],8400100+b)
        for b,a in [(True,'after'),(8,'none'),(0,'shuffled_before')]:
            with self.assertRaises(ValueError):m.cell_config(b,a)

    def test_assigned_snapshot_preserves_complete_original_record(self):
        m=self.module();source=Path(__file__).resolve().parents[2]/'data'
        if not source.exists():self.skipTest('Source archive is tested in host preflight')
        from agent.mea_stim_offline import StimRecording
        original=StimRecording(source,pin=m.PIN,phase='after',split='train')
        assigned=m.training_recording(source,m.PIN,m.cell_config(0,'shuffled_after'))
        order=m.train_order(0,'shuffled_after')
        for i in range(385):
            self.assertEqual(assigned.snapshot(i,'fixture').to_dict(),original.snapshot(order[i],'fixture').to_dict())
        self.assertEqual(assigned.summary()['assignment'],order)
        for i in [-1,385,445,True]:
            with self.assertRaises(ValueError):assigned.snapshot(i,'fixture')

    def test_paired_inference_zero_constant_and_invalid(self):
        m=self.module()
        zero=m.paired_stats([0.]*8,3);positive=m.paired_stats([1.]*8,3)
        self.assertEqual(zero['p_adjusted'],1.)
        self.assertEqual(positive['bootstrap_95'],[1.,1.])
        self.assertEqual(positive['p_two_sided'],2/256)
        self.assertEqual(positive['p_adjusted'],6/256)
        self.assertEqual(positive['positive_pairs'],8)
        for values in [[1.]*7,[float('nan')]*8]:
            with self.assertRaises(ValueError):m.paired_stats(values,3)

    def test_missing_duplicate_nonfinite_and_mixed_contexts_not_replication(self):
        m=self.module()
        self.assertEqual(m.analyze([])['status'],'incomplete')
        cells=[]
        for b in range(8):
            for a in m.PROTOCOL['arms']:
                fields={k:100. for k in m.ENDPOINTS};fields['terminated']=0.
                if a=='after':fields['visited_count']=101.
                cells.append({'block':b,'arm':a,'pre':{c:dict(fields) for c in ('before','after')},'post':{c:dict(fields) for c in ('before','after')}})
        result=m.analyze(cells)
        self.assertTrue(result['exploration_replicated'])
        self.assertTrue(result['temporal_order_evidence'])
        self.assertEqual(result['survival_family_size'],15)
        self.assertFalse(result['overall_r1_benefit'])
        self.assertEqual(result['survival_minimum_adjusted_p'],15*2/256)
        self.assertFalse(result['survival_significance_reachable'])
        with self.assertRaises(ValueError):m.analyze(cells+[cells[0]])
        cells[2]['post']['before']['visited_count']=float('nan')
        with self.assertRaises(ValueError):m.analyze(cells)

    def test_common_evaluation_and_forked_training_binding(self):
        self.module()
        spec=importlib.util.find_spec('tools.mea_shuffle_campaign')
        self.assertIsNotNone(spec,'Missing isolated campaign')
        c=importlib.import_module('tools.mea_shuffle_campaign')
        for context in ('before','after'):
            self.assertEqual(c.evaluation_schedule(1,'after',context),c.evaluation_schedule(1,'shuffled_after',context))
            self.assertEqual(c.evaluation_schedule(1,'after',context)['order'],list(range(445,596)))
        t=importlib.import_module('tools.mea_shuffle_train')
        self.assertIs(t.cell_config,self.module().cell_config)

    def test_same_gpu_block_schedule_and_incomplete_acceptance(self):
        spec=importlib.util.find_spec('tools.mea_shuffle_parallel')
        self.assertIsNotNone(spec,'Missing parallel completion gate')
        w=importlib.import_module('tools.mea_shuffle_parallel')
        self.assertEqual(w.arm_order(0),['none','before','after','shuffled_after'])
        orders=[w.arm_order(b) for b in range(8)]
        for position in range(4):
            self.assertEqual(sorted(order[position] for order in orders),sorted(['none','before','after','shuffled_after']*2))
        for status in [[],[{'passed':True,'block':b,'accepted_cells':4} for b in range(7)],
                       [{'passed':True,'block':0,'accepted_cells':4}]*8]:
            with self.assertRaises(ValueError):w.validate_blocks(status)
        status=[{'passed':True,'block':b,'accepted_cells':4} for b in range(8)]
        w.validate_blocks(status)
        status[0]['passed']=False
        with self.assertRaises(ValueError):w.validate_blocks(status)

    def test_stage_timeout_returns_failure_and_reaps_process_group(self):
        import sys,tempfile
        spec=importlib.util.find_spec('tools.mea_shuffle_parallel')
        self.assertIsNotNone(spec,'Missing bounded stage runner')
        w=importlib.import_module('tools.mea_shuffle_parallel')
        with tempfile.TemporaryDirectory() as t:
            result=w.run_stage([sys.executable,'-c','import time;time.sleep(5)'],Path(t)/'timeout.log',timeout=.1)
            self.assertEqual(result['exit_code'],-1)
            self.assertEqual(result['error'],'timeout')

if __name__=='__main__':unittest.main()
