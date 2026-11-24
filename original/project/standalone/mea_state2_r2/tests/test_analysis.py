import copy
import importlib
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'standalone/mea_state2_r2'),str(ROOT/'standalone/mea_binary_r2'),str(ROOT)]
REGISTRATION='1'*64


def cells():
    rows=[]
    for block in range(8):
        for arm,value in [('A0',0.),('A',1.),('B',2.),('C',3.),('D',3.)]:
            metric=dict(r1_sum=value,visited_count=10.,final_health=100.,final_hunger=90.,final_thirst=90.,elapsed_ticks=20.,terminated=0)
            rows.append(dict(block=block,arm=arm,passed=True,smoke=False,registration_sha256=REGISTRATION,
                             metrics={phase:dict(full=metric.copy(),first16=metric.copy()) for phase in ('after','before')}))
    return rows


class AnalysisTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('mea_state2_r2.analysis')
        except ModuleNotFoundError:self.fail('Registered outcome analysis missing')

    def test_known_paired_effect_and_single_primary(self):
        m=self.module();result=m.analyze(cells(),REGISTRATION)
        stat=result['contrasts']['C_minus_B']['r1_sum']
        self.assertEqual(stat['mean_difference'],1.)
        self.assertEqual(stat['bootstrap_95'],[1.,1.])
        self.assertEqual(stat['p_two_sided'],2/256)
        self.assertTrue(result['digital_state2_benefit_detected'])
        self.assertFalse(result['temporal_order_supported'])
        self.assertFalse(result['wetware_closed_loop_effect_verified'])

    def test_missing_failed_duplicate_and_nonfinite_rejected(self):
        m=self.module()
        self.assertEqual(m.analyze(cells()[:-1],REGISTRATION)['status'],'incomplete')
        bad=cells();bad[0]['passed']=False
        self.assertEqual(m.analyze(bad,REGISTRATION)['status'],'incomplete')
        with self.assertRaises(ValueError):m.analyze(cells()+[cells()[0]],REGISTRATION)
        bad=cells();bad[0]['metrics']['after']['full']['r1_sum']=float('nan')
        with self.assertRaises(ValueError):m.analyze(bad,REGISTRATION)

    def test_mixed_missing_registration_and_smoke_are_rejected(self):
        m=self.module()
        for mutation in ('mixed','missing','smoke'):
            bad=cells()
            if mutation=='mixed':bad[3]['registration_sha256']='2'*64
            elif mutation=='missing':bad[3].pop('registration_sha256')
            else:bad[3]['smoke']=True
            with self.assertRaises(ValueError):m.analyze(bad,REGISTRATION)

    def test_implicit_or_mixed_registration_cannot_be_accepted(self):
        m=self.module();bad=cells();bad[3]['registration_sha256']='2'*64
        with self.assertRaises(ValueError):m.analyze(bad)


if __name__=='__main__':unittest.main()
