import unittest
import numpy as np


def pairs():
    out=[]
    for i in range(10):
        first='true' if i<5 else 'sham'
        second='sham' if first=='true' else 'true'
        out.append({'status':'pass','pair':i+1,'trials':[
            {'id':9001+i*2+j,'pair':i+1,'arm':arm,'time_s':30.+60*j,'pre_rate_hz':0.,'post_rate_hz':10. if arm=='true' else 0.,'delta_hz':10. if arm=='true' else 0.}
            for j,arm in enumerate((first,second))]})
    return out


class AnalysisTests(unittest.TestCase):
    def test_only_delayed_2_to_5_window_enters_response(self):
        from mea_stim_v2.analysis import trial_response
        times=np.array([5.,7.999,8.,10.,11.,12.,14.999,15.])
        q=trial_response(times,np.zeros(len(times),dtype=int),[0],10.,20.)
        self.assertEqual(q['pre_rate_hz'],2/3)
        self.assertEqual(q['post_rate_hz'],2/3)
        self.assertEqual(q['delta_hz'],0)

    def test_252_balanced_assignments_and_direction_rules(self):
        from mea_stim_v2.analysis import summarize_pairs
        q=summarize_pairs(pairs())
        self.assertEqual(q['status'],'complete_qualified')
        self.assertEqual(q['randomization_assignments'],252)
        self.assertAlmostEqual(q['p_two_sided'],2/252)
        self.assertEqual(q['mean_difference_hz'],10.)
        self.assertTrue(q['repeat_response_candidate'])

    def test_incomplete_failed_or_unbalanced_has_no_primary_pvalue(self):
        from mea_stim_v2.analysis import summarize_pairs
        tests=[pairs()[:9],pairs(),pairs()]
        tests[1][4]['status']='failed'
        tests[2][5]['trials'][0]['time_s']=90.
        tests[2][5]['trials'][1]['time_s']=30.
        for data in tests:
            q=summarize_pairs(data)
            self.assertIsNone(q['p_two_sided'])
            self.assertFalse(q['repeat_response_candidate'])

    def test_short_trial_window_rejected(self):
        from mea_stim_v2.analysis import trial_response
        with self.assertRaises(ValueError):trial_response(np.array([]),np.array([]),[0],3.,20.)

    def test_duplicate_event_identity_or_trial_pair_never_earns_pvalue(self):
        from mea_stim_v2.analysis import summarize_pairs
        for kind in ('duplicate','pair'):
            data=pairs()
            if kind=='duplicate':data[3]['trials'][0]['id']=9001
            if kind=='pair':data[3]['trials'][0]['pair']=9
            self.assertIsNone(summarize_pairs(data)['p_two_sided'])


if __name__=='__main__':unittest.main()
