import unittest
import numpy as np
from analysis_math import trial_response, paired_summary, rate_bins, completion_issues, covered_rate
from protocol import SESSION_ID

class AnalysisTests(unittest.TestCase):
 def test_half_open_fixed_windows_and_roi(self):
  t=np.array([5,7.999,8,12,14.999,15,13]);c=np.array([1,1,1,1,1,1,2])
  r=trial_response(t,c,[1],10)
  self.assertEqual(r['before_count'],2);self.assertEqual(r['after_count'],2)
  self.assertEqual(r['delta_hz'],0)
 def test_balanced_pairs_exact_sign_flip(self):
  rows=[{'pair':p,'arm':a,'delta_hz':d} for p in range(1,7) for a,d in [('true',2),('sham',0)]]
  r=paired_summary(rows)
  self.assertEqual(r['mean_true_minus_sham_hz'],2)
  self.assertEqual(r['exact_sign_flip_two_sided_p'],2/64)
  with self.assertRaises(ValueError):paired_summary(rows[:-1])
 def test_actual_balanced_order_randomization_has_twenty_assignments(self):
  rows=[]
  for p in range(1,7):
   arms=['true','sham'] if p<=3 else ['sham','true']
   rows.extend({'pair':p,'arm':a,'delta_hz':2 if a=='true' else 0} for a in arms)
  r=paired_summary(rows)
  self.assertEqual(r['balanced_order_randomization_assignments'],20)
  self.assertEqual(r['balanced_order_randomization_two_sided_p'],.1)
 def test_partial_bin_exposure_not_false_drop(self):
  centers,rate=rate_bins(np.array([.5,9.5,10.1]),10.2,10)
  np.testing.assert_allclose(rate,[.2,5])
 def test_interrupted_stage_never_qualified(self):
  base={'status':'pass','stage':'baseline','session_id':SESSION_ID,'roi_channels':list(range(50))}
  state={'stage_history':[],'cleanup_errors':[],'r2_enabled':False,'training_enabled':False}
  self.assertIn('stage_not_recorded_as_fully_closed',completion_issues(state,base,'control'))
  state['stage_history']=[{'stage':'control','recording_closed_utc':'2026-11-17T05:20:00Z'}]
  self.assertEqual(completion_issues(state,base,'control'),[])
  base['status']='failed';self.assertIn('baseline_not_qualified',completion_issues(state,base,'control'))
 def test_rate_uses_actual_observation_exposure(self):
  r=covered_rate(np.array([61,599.2]),60,600,599.5)
  self.assertEqual(r['seconds'],539.5);self.assertAlmostEqual(r['hz'],2/539.5)

if __name__=='__main__':unittest.main()
