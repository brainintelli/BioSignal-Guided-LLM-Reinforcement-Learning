from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def invoke(self,*args):
        self.assertTrue((ROOT/'run.py').exists(),'entry point is not implemented')
        return subprocess.run([sys.executable,str(ROOT/'run.py'),*args],capture_output=True,text=True,encoding='utf-8')
    def test_default_preview_does_not_need_binding(self):
        r=self.invoke();self.assertEqual(r.returncode,0,r.stderr)
        data=json.loads(r.stdout);self.assertEqual(data['hardware_connected'],False)
        self.assertEqual(data['post_seconds'],1800)
    def test_isolated_python_preview_finds_only_bundled_modules(self):
        r=subprocess.run([sys.executable,'-I','-S',str(ROOT/'run.py')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(json.loads(r.stdout)['hardware_connected'])
    def test_unbound_template_run_rejects_before_sdk_network(self):
        r=self.invoke('run','--binding',str(ROOT/'binding.template.json'),'--output','/tmp/never_started')
        self.assertNotEqual(r.returncode,0);self.assertIn('Unbound',r.stderr)
    def test_simulation_success_and_gate_failure_are_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            for scenario,want,sends in [('success','completed',1),('baseline-failure','failed',0),('lost-reply','output_unknown',1)]:
                output=Path(d)/scenario
                r=self.invoke('simulate','--output',str(output),'--scenario',scenario)
                self.assertEqual(r.returncode,0,r.stderr)
                report=json.loads((output/'simulation.json').read_text())
                self.assertEqual(report['status'],want);self.assertEqual(report['nonzero_sequence_attempts'],sends)
                self.assertTrue(report['synthetic']);self.assertFalse(report['hardware_connected'])


if __name__=='__main__':unittest.main()
