import unittest
import numpy as np
from qc_math import baseline_gate,recovery_gate

class QcMathTests(unittest.TestCase):
 def args(self):
  return dict(channels=np.arange(100),counts_before=np.full(100,12),counts_after=np.full(100,12),
   seconds_before=100,seconds_after=100,sd_before=np.full(100,10.),sd_after=np.full(100,10.),
   median_before=np.full(100,512.),median_after=np.full(100,512.),
   rails_before=np.zeros(100),rails_after=np.zeros(100),excluded=list(range(6)))
 def test_fixed_exclusion_and_stable_gate(self):
  r=baseline_gate(**self.args());self.assertEqual(r['status'],'pass');self.assertEqual(r['roi_channels'],list(range(6,100)))
 def test_noise_and_loss_of_activity_fail(self):
  a=self.args();a['sd_after'][6:46]=30
  self.assertEqual(baseline_gate(**a)['status'],'failed')
  a=self.args();a['counts_after'][6:86]=0
  self.assertEqual(baseline_gate(**a)['status'],'failed')
 def test_recovery_requires_each_prespecified_window(self):
  self.assertTrue(recovery_gate(np.ones(100),np.ones((2,100)),np.zeros((2,100)))['passed'])
  after=np.ones((2,100));after[1,:10]=3
  self.assertFalse(recovery_gate(np.ones(100),after,np.zeros((2,100)))['passed'])

if __name__=='__main__':unittest.main()
