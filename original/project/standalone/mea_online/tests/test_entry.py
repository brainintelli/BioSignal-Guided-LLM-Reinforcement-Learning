import importlib
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from mea_online.common import clone,digest,read,write
from mea_online.config import Budget,template
from mea_online.offline import Clock,SyntheticSource,fixture_binding

def module(name):
    try:return importlib.import_module('mea_online.'+name)
    except ModuleNotFoundError as e:raise AssertionError('Required online module missing: '+name) from e

class CalibrationTests(unittest.TestCase):
    def test_fit_uses_60_baseline_and_30_independent_validation_windows(self):
        clock=Clock();b=fixture_binding();source=SyntheticSource(b,clock)
        with tempfile.TemporaryDirectory() as d:
            c=module('calibration').collect(source,b,Path(d)/'calibration',Budget(1000,clock),live=False,clock=clock)
            self.assertEqual(len(c['calibration_window_ids']),60);self.assertEqual(len(c['validation_window_ids']),30)
            self.assertEqual(len(set(c['calibration_window_ids']+c['validation_window_ids'])),90)
            self.assertEqual(c['source'],'simulation');self.assertFalse(source.closed)

    def test_fault_leaves_raw_evidence_but_no_calibration_pass(self):
        clock=Clock();b=fixture_binding();source=SyntheticSource(b,clock,fault_at=3)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calibration'
            with self.assertRaises(ConnectionError):module('calibration').collect(source,b,p,Budget(1000,clock),live=False,clock=clock)
            self.assertTrue((p/'windows.jsonl').exists());self.assertFalse((p/'calibration.json').exists())

    def test_holdout_shift_blocks_normalization(self):
        clock=Clock();b=fixture_binding();base=SyntheticSource(b,clock)
        class Shift:
            def capture(self,request_id):
                w=base.capture(request_id)
                if base.calls>60:
                    for e in w['events']:e[3]*=10000
                w['window_sha256']=digest({k:v for k,v in w.items() if k!='window_sha256'});return w
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'validation'):module('calibration').collect(Shift(),b,Path(d)/'cal',Budget(1000,clock),live=False,clock=clock)

class EntryTests(unittest.TestCase):
    def test_check_unbound_template_imports_no_runtime_or_socket(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';write(p,template())
            with patch('socket.socket.connect',side_effect=AssertionError('network')):
                self.assertEqual(module('cli').main(['check','--config',str(p)]),0)

    def test_unbound_live_rejected_before_claim_or_model(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';write(p,template());out=Path(d)/'run'
            with self.assertRaisesRegex(ValueError,'binding'):module('cli').main(['live','--config',str(p),'--output',str(out)])
            self.assertFalse(out.exists())

    def test_offline_cli_writes_real_end_to_end_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'verify'
            self.assertEqual(module('cli').main(['offline-verify','--output',str(p)]),0)
            result=read(p/'verification.json');self.assertTrue(result['passed']);self.assertFalse(result['real_mea']);self.assertFalse(result['real_m9'])
            self.assertTrue((p/'normal/run/result.json').exists())

    def test_supervisor_terminates_overdue_child(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(time.time()-7080+.3)
            r=module('supervisor').run([sys.executable,'-c','import time;time.sleep(30)'],Path(d),b,grace=.1)
            self.assertEqual(r['status'],'deadline_stopped');self.assertIsNotNone(r['returncode']);self.assertLess(r['elapsed_seconds'],5.)

if __name__=='__main__':unittest.main()
