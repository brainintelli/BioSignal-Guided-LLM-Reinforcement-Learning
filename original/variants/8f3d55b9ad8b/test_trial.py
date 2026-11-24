import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'BrainStimulation'))
from trial_guard import TrialGuard


class Gates(unittest.TestCase):
    def make(self, root, wall=1000, mono=30):
        return TrialGuard(root,started_wall=900,wall_now=wall,monotonic_now=mono)

    def test_unqualified_cannot_reserve(self):
        with tempfile.TemporaryDirectory() as root:
            g=self.make(root)
            with self.assertRaises(RuntimeError):g.reserve(wall_now=1000,monotonic_now=30,qualified=False)
            self.assertFalse(g.claim.exists())

    def test_persisted_exactly_15_once_across_restarts(self):
        with tempfile.TemporaryDirectory() as root:
            g=self.make(root);g.reserve(wall_now=1000,monotonic_now=30,qualified=True)
            self.assertEqual(json.loads(g.claim.read_text())['reserved_pulses'],15)
            with self.assertRaises(FileExistsError):
                self.make(root).reserve(wall_now=1000,monotonic_now=30,qualified=True)

    def test_uncertain_response_cannot_retry(self):
        with tempfile.TemporaryDirectory() as root:
            g=self.make(root)
            try:
                g.reserve(wall_now=1000,monotonic_now=30,qualified=True)
                raise TimeoutError('response lost')
            except TimeoutError:pass
            with self.assertRaises(FileExistsError):g.reserve(wall_now=1001,monotonic_now=31,qualified=True)

    def test_full_10_minutes_plus_closeout_required(self):
        with tempfile.TemporaryDirectory() as root:
            g=self.make(root)
            with self.assertRaises(RuntimeError):g.reserve(wall_now=7360,monotonic_now=6390,qualified=True)
            self.assertFalse(g.claim.exists())
            g.reserve(wall_now=7359,monotonic_now=6389,qualified=True)

    def test_clock_rollback_does_not_extend(self):
        with tempfile.TemporaryDirectory() as root:
            g=self.make(root)
            with self.assertRaises(RuntimeError):g.reserve(wall_now=1000,monotonic_now=6500,qualified=True)

    def test_expired_and_future_start(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):self.make(root,wall=8101)
            with self.assertRaises(ValueError):self.make(root,wall=800)

    def test_frozen_sdk_wire_sequence_without_any_connection(self):
        with patch.object(socket,'create_connection',side_effect=AssertionError('NETWORK FORBIDDEN')):
            from brain_stimulation.plan import StimulationParameters
            from brain_stimulation.runner import build_sequence
            p=StimulationParameters.from_params(json.loads((ROOT/'photo_parameters.template.json').read_text()))
            class Capture:
                def __init__(self):self.commands=['system_delay_samples 100']
                def append(self,obj):self.commands.append(obj.set())
            sequence=Capture();build_sequence(p,p.amplitude_schedule(),2.981901,sequence)
            time_steps=0;dac=[];events=[]
            for line in sequence.commands:
                parts=line.split()
                if parts[0]=='system_delay_samples':time_steps+=int(parts[1])
                elif parts[0]=='mea_set_dac':dac.append((time_steps,int(parts[2])))
                elif parts[0]=='system_trigger_event':events.append((time_steps,int(parts[3])))
                else:self.fail('Unexpected command '+line)
            starts=[100+b*4000+pulse*200 for b in range(5) for pulse in range(3)]
            self.assertEqual([t for t,code in dac if code==311],starts)
            self.assertEqual([code for t,code in dac],[311,713,512]*15)
            self.assertEqual([t for t,code in dac if code==713],[s+4 for s in starts])
            self.assertEqual([t for t,code in dac if code==512],[s+8 for s in starts])
            self.assertEqual([t for t,event in events],starts)
            self.assertEqual(len({event for t,event in events}),15)
            self.assertEqual(time_steps,16508)
            self.assertAlmostEqual(p.stimulation_seconds(),.8254)
            (ROOT/'sequence_preview.json').write_text(json.dumps({'status':'software_preview_only',
                'no_network_connections':True,'pulse_onset_ms':[s*.05 for s in starts],
                'codes':[311,713,512],'nominal_phase_amplitude_mv':201*2.981901,
                'total_sequence_seconds':time_steps*.00005,'serialized_commands':sequence.commands},indent=2))


if __name__=='__main__':unittest.main()
