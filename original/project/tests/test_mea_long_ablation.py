import copy
import importlib
import unittest


class LongAblationTests(unittest.TestCase):
    def m(self):
        return importlib.import_module('agent.mea_long_ablation')

    def test_protocol_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('agent.mea_long_ablation'))

    def test_split_derangements_have_no_leakage_or_duplicate(self):
        m=self.m()
        for b in range(8):
            for split,lo,hi in [('train',0,1024),('eval',1085,1685)]:
                real=m.assignment('real',split,b);shuffled=m.assignment('shuffled',split,b)
                self.assertEqual(real,list(range(lo,hi+1)))
                self.assertEqual(shuffled[-1],hi)
                self.assertEqual(set(shuffled),set(real))
                self.assertTrue(all(x!=y for x,y in zip(real[:-1],shuffled[:-1])))
                self.assertEqual(shuffled,m.assignment('shuffled',split,b))
        for args in [('real','bad',0),('real','train',8),('real','train',True),('bad','train',0)]:
            with self.assertRaises(ValueError):m.assignment(*args)

    def test_same_initialization_different_paired_sampling(self):
        m=self.m()
        configs=[m.cell_config(b,a) for b in range(8) for a in m.ARMS]
        self.assertEqual(len({c['initial_seed'] for c in configs}),1)
        for b in range(8):
            pair=[c for c in configs if c['block']==b]
            for key in ['world_seed','action_seed','ppo_seed']:
                self.assertEqual(len({c[key] for c in pair}),1)
        self.assertEqual(len({c['world_seed'] for c in configs}),8)

    def test_budget_and_mask_gate(self):
        m=self.m()
        for arm in m.ARMS:
            m.validate_budget({'actor':2,'critic':16})
            self.assertTrue(m.columns_expected(arm,arm!='none'))
            self.assertFalse(m.columns_expected(arm,arm=='none'))
        for steps in [{'actor':1,'critic':16},{'actor':2,'critic':15},{'actor':True,'critic':16}]:
            with self.assertRaises(ValueError):m.validate_budget(steps)

    def test_analysis_requires_all_cells_and_two_controls(self):
        m=self.m();cells=[]
        for b in range(8):
            for arm in m.ARMS:
                cells.append(dict(block=b,arm=arm,pre=0.,post=1. if arm=='real' else 0.))
        self.assertEqual(m.analyze(cells[:-1])['status'],'incomplete')
        self.assertTrue(m.analyze(cells)['positive_evidence'])
        equal=copy.deepcopy(cells)
        for c in equal:c['post']=0.
        self.assertFalse(m.analyze(equal)['positive_evidence'])
        with self.assertRaises(ValueError):m.analyze(cells+[cells[0]])

    def test_driver_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('tools.mea_long_ablation'))

    def test_evidence_receipt_is_bound_to_cell_source_and_content(self):
        from tools.mea_long_ablation import validate_receipt
        good=dict(passed=True,block=2,arm='real',registration_sha256='r',cell_manifest_sha256='m',
            source_manifest_sha256=self.m().PIN,training_decisions=1024,
            eval_model_errors={'pre':0.,'post':0.},fresh_process=True)
        validate_receipt(good,2,'real','r','m')
        for mutation in [{'arm':'none'},{'block':3},{'passed':False},{'fresh_process':False},
            {'registration_sha256':'x'},{'cell_manifest_sha256':'x'},
            {'source_manifest_sha256':'x'},{'training_decisions':32},
            {'eval_model_errors':{'pre':0.}},{'eval_model_errors':{'pre':0.,'post':float('nan')}}]:
            with self.assertRaises(ValueError):validate_receipt(good|mutation,2,'real','r','m')

    def test_none_masks_input_but_preserves_telemetry(self):
        from tools.mea_long_train import prepare
        class Data:
            def snapshot(self,i,context):return ('telemetry',i)
        class Policy:
            arm='none'
            def prepare(self,observations,snapshots):
                self.inputs=snapshots;return ['prepared']
        from tests.test_mea_p0 import observation
        p=Policy();s,row=prepare(p,Data(),7,observation())
        self.assertEqual(s,('telemetry',7));self.assertEqual(row,'prepared');self.assertEqual(p.inputs,[None])

    def test_assigned_recording_rejects_holdout_during_training(self):
        m=self.m()
        class Data:
            window_count=1797
            def snapshot(self,i,context):return i
            def summary(self):return {'source':'original'}
        assigned=m.AssignedRecording(Data(),m.cell_config(0,'shuffled'))
        self.assertEqual(assigned.snapshot(1024,'c'),1024)
        self.assertEqual(assigned.summary(),{'source':'original'})
        for index in [-1,1025,1085,True]:
            with self.assertRaises(ValueError):assigned.snapshot(index,'c')

    def test_substituted_critic_cannot_become_registered_original(self):
        m=importlib.import_module('tools.mea_long_ablation')
        self.assertTrue(hasattr(m,'validate_common_initial'),'original critic provenance gate is missing')
        expected=self.m().ORIGINAL_TENSORS
        m.validate_common_initial(expected)
        for key in ('actor','critic','base','reference'):
            with self.assertRaises(ValueError):m.validate_common_initial(expected|{key:'0'*64})

if __name__=='__main__':unittest.main()
