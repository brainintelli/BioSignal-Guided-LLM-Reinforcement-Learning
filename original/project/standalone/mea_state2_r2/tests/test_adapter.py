import copy
import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'standalone/mea_state2_r2'), str(ROOT/'standalone/mea_binary_r2'), str(ROOT)]
from mea_state2_r2.data import Dataset
from mea_state2_r2.features import digest, validate_snapshot
from agent.mea_contract import observation_key
from tests.test_mea_p0 import observation


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT/'output/mea_28080_shuffle_20261117_v2/data'
        cls.data = Dataset(source if source.exists() else ROOT.parent/'data')

    def test_actor_critic_share_vector_and_masked_arms_share_format(self):
        try:
            m = importlib.import_module('mea_state2_r2.adapter')
        except ModuleNotFoundError:
            self.fail('New adapter not implemented')
        obs = observation()
        s = self.data.snapshot(0, 'after', observation_key(obs), 'C', 0, 'train')
        c = m.encode(obs, s, 'C', self.data.normalization_id)
        a = m.encode(obs, s, 'A', self.data.normalization_id)
        b = m.encode(obs, s, 'B', self.data.normalization_id)
        self.assertEqual(c['critic_input'][-4:], s['normalized'])
        self.assertIn(str(s['normalized'][0]), c['messages'][1]['content'])
        self.assertEqual(a['critic_input'][-4:], [0.,0.,0.,0.])
        self.assertEqual(a['messages'], b['messages'])
        self.assertEqual(len(c['critic_input']), 112)
        bad = copy.deepcopy(s)
        bad['schema'] = 'mea.features.p0.v1'
        with self.assertRaises(ValueError): m.encode(obs,bad,'C',self.data.normalization_id)
        with self.assertRaises(ValueError): m.encode(obs,s,'C','wrong-transform')

    def test_finite_splits_and_snapshot_context(self):
        d = self.data
        self.assertEqual(d.normalization['fit_range'], [0,385])
        self.assertEqual(d.snapshot(0,'after','ctx','C',0,'eval')['window_index'],445)
        with self.assertRaises(ValueError): d.snapshot(385,'after','ctx','C',0,'train')
        with self.assertRaises(ValueError): d.snapshot(0,'before','ctx','C',0,'train')
        s=d.snapshot(0,'after','ctx','D',0,'train')
        with self.assertRaises(ValueError): validate_snapshot(s,'other',d.normalization_id)
        s['normalized'][0]=.999
        with self.assertRaises(ValueError): validate_snapshot(s,'ctx',d.normalization_id)


class TrajectoryTests(unittest.TestCase):
    def module(self):
        try: return importlib.import_module('mea_state2_r2.trajectory')
        except ModuleNotFoundError: self.fail('Trajectory acceptance not implemented')

    def test_mid_and_exact_end_terminal_cut_bootstrap(self):
        m=self.module()
        rows=[dict(reward=1.,value=0.,duration=1,terminal=True),
              dict(reward=2.,value=0.,duration=1,terminal=False)]
        m.finish(rows,3.,gamma=.5,lam=1.)
        self.assertEqual([r['advantage'] for r in rows],[1.,3.5])
        self.assertEqual(rows[0]['next_value'],0.)
        rows[-1]['terminal']=True
        m.finish(rows,999.,gamma=.5,lam=1.)
        self.assertEqual([r['advantage'] for r in rows],[1.,2.])
        self.assertEqual(rows[-1]['next_value'],0.)

    def test_duration_discount_and_independent_audit(self):
        m=self.module()
        rows=[dict(reward=1.,value=.5,duration=2,terminal=False),
              dict(reward=2.,value=1.,duration=1,terminal=False)]
        m.finish(rows,3.,gamma=.5,lam=1.)
        self.assertAlmostEqual(rows[0]['advantage'],1.375)
        m.audit(rows,3.,gamma=.5,lam=1.)
        rows[0]['advantage']+=.1
        with self.assertRaises(ValueError):m.audit(rows,3.,gamma=.5,lam=1.)


if __name__=='__main__': unittest.main()
