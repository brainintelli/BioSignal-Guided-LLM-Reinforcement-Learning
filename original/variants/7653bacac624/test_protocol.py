import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from mea_stim_v2.config import Config, ConfigurationError
from mea_stim_v2.protocol import Budget, BudgetError, schedule, pulse_codes
from mea_stim_v2.journal import Ledger

class Clock:
 def __init__(self): self.w=100000.;self.m=50.
 def wall(self):return self.w
 def mono(self):return self.m
 def sleep(self,t):self.w+=t;self.m+=t

class ProtocolTests(unittest.TestCase):
 def test_unbound_or_r2_config_never_admits_live(self):
  with self.assertRaises(ConfigurationError):Config.from_dict({'r2_enabled':True})
  c=Config.template()
  with self.assertRaises(ConfigurationError):c.validate_live()
  self.assertFalse(c.r2_enabled);self.assertFalse(c.training_enabled)
 def test_true_and_sham_codes_are_finite_single_pulses(self):
  self.assertEqual(pulse_codes('true',2.981901),[('dac',412),('delay',4),('dac',612),('delay',4),('dac',512)])
  self.assertEqual(pulse_codes('sham',2.981901),[('dac',512),('delay',4),('dac',512),('delay',4),('dac',512)])
  for x in [0,float('nan'),float('inf'),.0001]:
   with self.assertRaises(ValueError):pulse_codes('true',x)
 def test_pair_design_and_registration_reproduce(self):
  s=schedule(42);self.assertEqual(s,schedule(42));self.assertEqual(len(s),10)
  self.assertEqual(sum(p['events'][0]['arm']=='true' for p in s),5)
  self.assertEqual([e['id'] for p in s for e in p['events']],list(range(9001,9021)))
  self.assertTrue(all(sorted(e['arm'] for e in p['events'])==['sham','true'] and 45<=p['interval_s']<=75 for p in s))
 def test_budget_never_extends_and_cutoff_is_distinct(self):
  c=Clock();b=Budget(100000,c);c.sleep(6200);b.require(900)
  with self.assertRaises(BudgetError):b.admit_pair()
  c.w-=2000
  self.assertAlmostEqual(b.remaining(),1000)
  c.sleep(101)
  with self.assertRaises(BudgetError):b.before_output(1)
 def test_entry_boundary_does_not_borrow_recovery(self):
  c=Clock();b=Budget(100000,c);c.sleep(6165);b.admit_pair();c.sleep(.001)
  with self.assertRaises(BudgetError):b.admit_pair()
 def test_reserved_lost_reply_dose_and_global_launch_are_not_reused(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);l=Ledger(root/'registry','org-1',{'started':100000},root/'run')
   l.reserve(8202,'true',100001);self.assertEqual(l.totals(),{'true':1,'sham':0})
   with self.assertRaises(FileExistsError):l.reserve(8202,'true',100002)
   with self.assertRaises(FileExistsError):Ledger(root/'registry','org-1',{'started':200000},root/'other')
   for i in range(10):l.reserve(9000+i,'true',100003+i)
   with self.assertRaises(RuntimeError):l.reserve(9999,'true',100020)
   self.assertEqual(l.totals()['true'],11)

if __name__=='__main__':unittest.main()
