import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mea_live_22554_20261117/event_fixed_package/BrainStimulation'))
from brain_stimulation import maxlab as mx
from capture_intake import allowed_commands

class IntakeTest(unittest.TestCase):
 def test_only_zero_and_power_off_hardware_are_allowed(self):
  allowed=allowed_commands(mx,Path('/tmp/session/intake'))
  self.assertIn('mea_set_dac 0 512 512',allowed)
  self.assertNotIn('mea_set_dac 0 412 512',allowed)
  for u in range(32):
   self.assertIn(mx.StimulationUnit(u).power_up(False).connect(False).set(),allowed)
   self.assertNotIn(mx.StimulationUnit(u).power_up(True).connect(True).set(),allowed)
  self.assertFalse(any(x.startswith(('sequence','system_trigger','mea_array','mea_initialize')) for x in allowed))
if __name__=='__main__':unittest.main()
