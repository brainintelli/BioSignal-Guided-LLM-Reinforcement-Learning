import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'standalone/mea_state2_r2'),str(ROOT/'standalone/mea_binary_r2'),str(ROOT)]


class IntegrityTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('mea_state2_r2.integrity')
        except ModuleNotFoundError:self.fail('Independent identity and summary acceptance not implemented')

    def test_pinned_assets_reject_changed_external_file(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/'assets';assets.mkdir();p=assets/'tokenizer.json';p.write_bytes(b'original')
            spec=dict(schema='mea.external.assets.v1',roots={'runtime':str(assets)},
                      files={'runtime/tokenizer.json':dict(size=8,sha256=hashlib.sha256(b'original').hexdigest())})
            (root/'external_assets.json').write_text(json.dumps(spec))
            self.assertEqual(m.verify_assets(root)['files_verified'],1)
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):m.verify_assets(root)

    def test_frozen_baseline_requires_original_critic_too(self):
        m=self.module()
        from agent.mea_long_ablation import ORIGINAL_TENSORS
        actual=dict(ORIGINAL_TENSORS,critic='raw-108-dimensional-hash')
        m.check_baseline_identity(actual,ORIGINAL_TENSORS['critic'])
        with self.assertRaises(ValueError):m.check_baseline_identity(actual,'wrong-critic')

    def test_changed_round_counts_and_pre_metrics_fail_acceptance(self):
        m=self.module()
        rows=[dict(terminal=False,r1_score={'reward':.1},reward_record={'r2_raw':1,'r2_contribution':.001})]
        report=dict(decisions=1,positive=1,negative=0,terminals=0,r1_sum=.1,r2_contribution_sum=.001,
                    applied_steps={'actor':2,'critic':16})
        m.check_round_summary(report,rows,{'actor':2,'critic':16})
        bad=dict(report,positive=9)
        with self.assertRaises(ValueError):m.check_round_summary(bad,rows,{'actor':2,'critic':16})
        train=dict(passed=True,rounds=[report])
        pre={'full':{'r1_sum':.25}}
        m.check_mirrors(train,[report],pre,pre)
        bad=copy.deepcopy(train);bad['rounds'][0]['positive']=9
        with self.assertRaises(ValueError):m.check_mirrors(bad,[report],pre,pre)
        with self.assertRaises(ValueError):m.check_mirrors(train,[report],{'full':{'r1_sum':1}},pre)


if __name__=='__main__':unittest.main()
