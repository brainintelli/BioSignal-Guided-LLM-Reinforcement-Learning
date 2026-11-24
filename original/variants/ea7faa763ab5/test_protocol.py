from pathlib import Path
import tempfile,json,unittest
from protocol import Budget, make_control_schedule, validate_gate, sequence_commands


class ProtocolTests(unittest.TestCase):
 def budget(self,path,wall=1000,mono=20):
  return Budget(Path(path),deadline_wall=5000,wall_now=wall,mono_now=mono,prior_true=5)
 def test_balanced_pairs_are_fixed_and_not_all_one_order(self):
  s=make_control_schedule();self.assertEqual(len(s),12)
  self.assertEqual(sum(x['arm']=='true' for x in s),6)
  self.assertEqual(sum(s[i]['arm']=='true' for i in range(0,12,2)),3)
  for i in range(0,12,2):self.assertEqual({s[i]['arm'],s[i+1]['arm']},{'true','sham'})
  self.assertEqual(s,make_control_schedule())
 def test_dose_reserved_before_send_never_reused(self):
  with tempfile.TemporaryDirectory() as d:
   b=self.budget(d);b.reserve('pilot',1,1,45.0054,1100,120)
   with self.assertRaises(FileExistsError):b.reserve('pilot',1,1,45.0054,1100,120)
   b2=self.budget(d);self.assertEqual(b2.totals(),(6,1))
   b2.reserve('control',6,6,495.0054,1200,220);self.assertEqual(b2.totals(),(12,7))
 def test_caps_and_recovery_and_clock_rollback(self):
  with tempfile.TemporaryDirectory() as d:
   b=self.budget(d)
   with self.assertRaises(RuntimeError):b.reserve('too_many',21,1,1,1100,120)
   with self.assertRaises(RuntimeError):b.reserve('too_many_shams',1,13,1,1100,120)
   with self.assertRaises(RuntimeError):b.reserve('late',1,1,45.0054,4215,3235)
   with self.assertRaises(RuntimeError):b.reserve('connect_plus_reply_margin',1,1,45.0054,4210,3230)
   with self.assertRaises(RuntimeError):b.reserve('rollback',1,1,45.0054,1100,3235)
   self.assertEqual(b.totals(),(5,0))
 def test_failed_or_mismatched_gate_cannot_authorize(self):
  q={'status':'pass','session_id':'22554-20261117T032900Z','stage':'baseline','roi_channels':list(range(50))}
  validate_gate(q,'baseline')
  for change in ({'status':'failed'},{'stage':'pilot'},{'session_id':'new'},{'roi_channels':[1]}):
   with self.assertRaises(ValueError):validate_gate({**q,**change},'baseline')
 def test_real_and_sham_timing_and_voltage(self):
  s=make_control_schedule();commands=sequence_commands(s,2.981901)
  ticks=0;onsets=[];dacs=[]
  for c in commands:
   if c.startswith('system_delay_samples'):
    n=int(c.split()[-1]);self.assertLessEqual(n,65535);ticks+=n
   elif c.startswith('system_trigger_event'):onsets.append(ticks)
   elif c.startswith('mea_set_dac'):dacs.append(int(c.split()[2]))
  self.assertEqual(onsets,[100+i*900000 for i in range(12)])
  self.assertEqual(ticks,9900108)
  for i,tr in enumerate(s):self.assertEqual(dacs[i*3:i*3+3],[412,612,512] if tr['arm']=='true' else [512]*3)

if __name__=='__main__':unittest.main()
