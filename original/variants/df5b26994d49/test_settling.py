"""Exercise actual runner requests against a localhost-only simulated server."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixture',Path('./github_program/BrainDesktop/tests/unit/test_standalone_stimulation.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
sys.path.insert(0,str(ROOT/'settled_package/BrainStimulation'))
from brain_stimulation.config import load_config
from brain_stimulation.runner import run_experiment
from brain_stimulation.transport import Endpoint


class SettlingTests(unittest.TestCase):
    def run_fixture(self,interrupt_stage=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);server=fixture.FakeMaxLab();clock=[0.0];events=[]
            reply=server.reply
            def observed(command):
                events.append((clock[0],command));return reply(command)
            server.reply=observed
            def wait(seconds):
                if interrupt_stage=='offset' and any(c=='system_offset' for t,c in events) and not any(c.startswith('saving_start_file') for t,c in events):
                    raise KeyboardInterrupt('deadline during calibration settling')
                clock[0]+=seconds
            try:
                params=load_config(ROOT/'settled_package/BrainStimulation/example.json')
                code=run_experiment(params,Endpoint('127.0.0.1',server.server_address[1],1),root,wait=wait)
                return code,events
            finally:server.close()

    def test_recording_starts_after_calibration_and_connection_have_settled(self):
        code,events=self.run_fixture()
        self.assertEqual(code,0)
        start=next(t for t,c in events if c=='saving_start_recording')
        offset=next(t for t,c in events if c=='system_offset')
        enabled=[t for t,c in events if c.startswith('mea_set_stimulation_unit ') and c.split()[2:4]==['1','1']]
        self.assertGreaterEqual(start-offset,10,'Recording overlaps offset calibration/settling')
        self.assertGreaterEqual(start-max(enabled),5,'Recording overlaps output-connection settling')

    def test_deadline_during_settling_cleans_up_without_recording_or_stimulus(self):
        code,events=self.run_fixture('offset')
        commands=[c for t,c in events]
        self.assertNotEqual(code,0,'Deadline interruption must abort the run')
        self.assertFalse(any(c.startswith('sequence_send ') for c in commands))
        self.assertNotIn('saving_start_recording',commands)
        self.assertIn('mea_set_dac 0 512 512',commands)


if __name__=='__main__':unittest.main()
