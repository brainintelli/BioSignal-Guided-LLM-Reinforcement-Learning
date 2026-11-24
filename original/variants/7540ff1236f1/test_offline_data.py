"""Synthetic unit fixtures only; never a biological/training source."""
import copy
import unittest

from sr_data import clean_windows, fit_state_profile, normalize, select_partitions
from sr_protocol import ARMS, PROTOCOL, assignment, reward_record


def windows(count=500):
    return [dict(source_id='unit_fixture_only', start_frame=i*20000,
                 end_frame=(i+1)*20000, spike_count=i % 13,
                 rate_hz_per_channel=(i % 13)/2,
                 active_channel_fraction=(i % 3)/2,
                 mean_abs_amplitude_uv=float(i % 7)) for i in range(count)]


def partitions():
    return [dict(split=s,source_id='unit_fixture_only',start_frame=a*20000,end_frame=b*20000)
            for s,a,b in [('calibration',0,60),('train',60,189),('evaluation',199,296),('holdout',306,350)]]


class OfflineDataTests(unittest.TestCase):
    def test_stimulus_masks_whole_window_and_exact_two_second_boundary(self):
        rows=windows(12)
        clean,excluded=clean_windows(rows, [{'start_frame':40001,'end_frame':40011}],[],20000)
        self.assertEqual([r['start_frame']//20000 for r in excluded],[2,3,4])
        self.assertEqual([r['start_frame']//20000 for r in clean],[0,1,5,6,7,8,9,10,11])
        clean,excluded=clean_windows(rows,[{'start_frame':40000,'end_frame':40000}],[],20000)
        self.assertEqual([r['start_frame']//20000 for r in excluded],[2,3])

    def test_qc_overlap_rejects_entire_window(self):
        clean,excluded=clean_windows(windows(6),[],[{'start_frame':19999,'end_frame':20001,'reason':'unit-test-mask'}],20000)
        self.assertEqual(len(clean),4)
        self.assertEqual([r['start_frame'] for r in excluded],[0,20000])

    def test_prior_recording_stimulus_tail_masks_next_recording(self):
        next_recording=windows(6)[3:]
        clean,excluded=clean_windows(next_recording,[{'start_frame':59000,'end_frame':60000}],[],20000)
        self.assertEqual([r['start_frame'] for r in excluded],[60000,80000])
        self.assertEqual([r['start_frame'] for r in clean],[100000])

    def test_budget_is_fixed_and_never_recycles(self):
        selected=select_partitions(windows(),partitions(),20000)
        self.assertEqual(len(selected),286)
        self.assertEqual([sum(r['split']==s for r in selected) for s in ('calibration','train','evaluation')],[60,129,97])
        self.assertTrue(all(r['start_frame']<296*20000 for r in selected))
        insufficient=[r for r in windows() if r['start_frame']!=70*20000]
        with self.assertRaisesRegex(ValueError,'insufficient train'):
            select_partitions(insufficient,partitions(),20000)

    def test_disjoint_chronological_partitions_and_embargo(self):
        bad=partitions();bad[2]['start_frame']=189*20000
        with self.assertRaisesRegex(ValueError,'embargo'):
            select_partitions(windows(),bad,20000)
        bad=partitions();bad[1]['start_frame']=59*20000
        with self.assertRaisesRegex(ValueError,'overlap'):
            select_partitions(windows(),bad,20000)

    def test_holdout_requires_at_least_one_complete_clean_window(self):
        no_holdout=[r for r in windows() if not 306*20000<=r['start_frame']<350*20000]
        with self.assertRaisesRegex(ValueError,'holdout'):select_partitions(no_holdout,partitions(),20000)
        short=partitions();short[-1]['end_frame']=short[-1]['start_frame']+19999
        with self.assertRaisesRegex(ValueError,'holdout'):select_partitions(windows(),short,20000)
        exactly_one=partitions();exactly_one[-1]['end_frame']=exactly_one[-1]['start_frame']+20000
        self.assertEqual(len(select_partitions(windows(),exactly_one,20000)),286)

    def test_normalization_fit_uses_only_calibration(self):
        selected=select_partitions(windows(),partitions(),20000)
        before=fit_state_profile(selected)
        for r in selected[60:]:r['spike_count']=999999
        self.assertEqual(before,fit_state_profile(selected))
        self.assertTrue(all(-5<=v<=5 for v in normalize(selected[60],before)))
        selected[0]['spike_count']=float('nan')
        with self.assertRaises(ValueError):fit_state_profile(selected)

    def test_every_reward_path_is_r1_only(self):
        self.assertEqual(ARMS,('baseline','state','shuffled'))
        self.assertEqual(PROTOCOL['aux_weight'],0.)
        for arm in ARMS:
            for training in (True,False):
                rr=reward_record(2.5,{'p':1.,'u':1.},arm,training=training)
                self.assertEqual(rr['total_reward'],2.5)
                self.assertEqual(rr['r1_weight'],1.)
                self.assertEqual(rr['aux_weight'],0.)
                self.assertIs(rr['auxiliary_enabled'],False)
                self.assertIs(rr['r2_enabled'],False)
                self.assertIs(rr['aux_enabled'],False)
                self.assertEqual(rr['reward_contract'],'r1_only_v1')

    def test_shuffle_never_leaks_split_or_bootstrap(self):
        for b in range(8):
            for split,n in [('train',128),('eval',96)]:
                normal=assignment('state',split,b);shuffled=assignment('shuffled',split,b)
                self.assertEqual(len(normal),n+1)
                self.assertEqual(set(normal),set(shuffled))
                self.assertEqual(normal[-1],shuffled[-1])
                self.assertTrue(all(a!=c for a,c in zip(normal[:-1],shuffled[:-1])))


if __name__=='__main__':unittest.main()
