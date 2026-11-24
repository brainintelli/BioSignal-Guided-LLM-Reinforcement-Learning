"""Whole sessions against a simulated rig that writes real-layout H5 files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import build_library
import quicklook
from oddball import design, library
from oddball.session import Session, Stop, write_once
from support import SESSION, SITE_CHANNELS, Clock, FakeRig


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'oddball-test'
        self.out.mkdir()
        self.clock = Clock()
        self.plan = design.plan(31)
        self.lines = []

    def session(self, age_min=10., **rig):
        self.rig = FakeRig(self.out, self.clock, **rig)
        start = self.clock.wall() - age_min * 60
        # run_session.run writes these two before the session starts.
        write_once(self.out / 'plan.json', self.plan)
        write_once(self.out / 'binding.json', dict(session_id=SESSION, chip_id='chip-test',
                                                   threshold_h5_dataset='settings/spike_threshold'))
        return Session(self.rig, self.out, self.plan, start, SESSION, [], self.clock,
                       say=lambda text, flush=False: self.lines.append(text))

    def read(self, name):
        return json.loads((self.out / name).read_text(encoding='utf-8'))

    def ledger(self):
        path = self.out / 'dose.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def test_complete_session_then_offline_library(self):
        session = self.session()
        status = session.run()
        self.assertEqual(status['outcome'], 'complete', status.get('error'))
        self.assertEqual(status['dose'], 60 + 1080)
        self.assertEqual(self.read('decision.json')['amplitude_mv'], 300)
        self.assertTrue(self.read('decision.json')['qualified'])
        self.assertFalse((self.out / 'screen-400.json').exists())
        names = [b['name'] for b in self.plan['blocks'] if b['stage'] != 'screen' or b['amplitude_mv'] == 300]
        self.assertEqual(self.rig.files, ['warmup'] + names + ['recovery'])
        self.assertTrue(all(b['passed'] for b in status['blocks']))
        self.assertTrue(self.rig.cleaned)
        self.assertEqual(status['cleanup'], [])
        # No real pulse after minute 95 and the whole session inside two hours.
        rows = self.ledger()
        self.assertEqual(rows[-1]['total'], 1140)
        self.assertLess(max(r['age_s'] for r in rows), design.STIM_CUTOFF_S)
        self.assertLess(status['age_s'], design.HARD_STOP_S)
        self.assertEqual(self.read('recovery.json')['events'], 0)
        self.assertGreaterEqual(self.read('recovery.json')['seconds'], 599)
        # Units are connected for every stimulation file, including sham trials, and off at the end.
        self.assertFalse(self.rig.on)
        self.assertEqual([on for name, on in self.rig.unit_log if name == 'main_r1_uniform'], [True, False])
        # Chunks leave the host every 20 s inside a block.
        sent = self.read('main_r1_uniform-sent.json')['chunks']
        self.assertEqual(len(sent), 10)
        gaps = [b['wall'] - a['wall'] for a, b in zip(sent, sent[1:])]
        self.assertTrue(all(abs(g - 20) < .05 for g in gaps))
        session.manifest()
        self.assertEqual(len(self.read('manifest.json')), 1 + 1 + 2 + 12 + 1)
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            quicklook.show(self.out)
        self.assertIn('screen 300 mV (actual 298.19): proceed True', shown.getvalue())
        self.assertIn('real pulses booked: 1140', shown.getvalue())
        self.assertIn('ended: complete', shown.getvalue())
        self.assertEqual(sum(line.startswith('main_r') for line in shown.getvalue().splitlines()), 12)

        with contextlib.redirect_stdout(io.StringIO()) as printed:
            report = build_library.main(self.out, self.out / 'analysis', False)
        self.assertIn('training allowed: True', printed.getvalue())
        self.assertTrue(report['raw_rechecked'])
        self.assertEqual(sum(b['passed'] for b in report['blocks']), 12)
        self.assertTrue(report['gates']['training_allowed'])
        made = library.check(json.loads((self.out / 'analysis' / 'library.json').read_text()))
        for site in design.SITES:
            means = made['stats'][site]['bin_means']
            self.assertGreater(means['low'], means['high'])
        psth = json.loads((self.out / 'analysis' / 'psth.json').read_text())
        real, sham = psth['spikes_per_trial']['H'], psth['spikes_per_trial']['Z']
        after = [i for i, t in enumerate(psth['bin_start_ms']) if 20 <= t < 240]
        self.assertGreater(sum(real[i] for i in after), sum(sham[i] for i in after) + 5)
        with self.assertRaises(FileExistsError):
            build_library.main(self.out, self.out / 'analysis', True)
        with contextlib.redirect_stdout(io.StringIO()):
            trusted = build_library.main(self.out, self.out / 'analysis-trusted', True)
        self.assertEqual(trusted['gates'], report['gates'])

    def test_silent_sample_escalates_once_then_gates_refuse_training(self):
        session = self.session(gain=0.)
        original = session.block
        def block(item, bits):
            result = original(item, bits)
            if sum(b['name'].startswith('main') for b in session.status['blocks']) == 2:
                (self.out / 'STOP').write_text('operator')
            return result
        session.block = block
        status = session.run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertIn('STOP', status['stop_reason'])
        self.assertFalse(self.read('screen-300.json')['proceed'])
        self.assertTrue(self.read('screen-400.json')['proceed'])
        decision = self.read('decision.json')
        self.assertEqual((decision['amplitude_mv'], decision['bits'], decision['qualified']), (400, 134, False))
        self.assertEqual(status['dose'], 60 + 60 + 180)
        self.assertIn('recorded', status['recovery'])
        self.assertTrue(self.rig.cleaned)
        with contextlib.redirect_stdout(io.StringIO()):
            report = build_library.main(self.out, self.out / 'analysis', False)
        self.assertFalse(report['gates']['training_allowed'])
        self.assertFalse((self.out / 'analysis' / 'library.json').exists())

    def test_output_during_the_zero_amplitude_rehearsal_stops_before_any_pulse(self):
        def leak(name, pulses, patches, events):
            if name == 'rehearsal':
                patches.append((SITE_CHANNELS['H'], events[3][0], events[3][0] + 800, 1023))
        status = self.session(tamper=leak).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertEqual(status['dose'], 0)
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal'])
        self.assertNotIn('recovery', status)
        self.assertTrue(self.rig.cleaned)

    def test_missing_onsets_stop_stimulation_and_still_record_recovery(self):
        def lose(name, pulses, patches, events):
            if name == 'screen300_1':
                del patches[:5]
        status = self.session(tamper=lose).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertIn('screen300_1', status['stop_reason'])
        self.assertEqual(status['dose'], 30)
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300_1', 'recovery'])
        self.assertFalse((self.out / 'decision.json').exists())

    def test_saturated_readout_channels_are_removed_before_the_main_blocks(self):
        def saturate(name, pulses, patches, events):
            if name.startswith('screen'):
                patches.extend((c, p['frame'] + 250, p['frame'] + 450, 0) for p in pulses for c in (55, 56))
        session = self.session(tamper=saturate)
        original = session.screen
        def screen():
            bits = original()
            (self.out / 'STOP').write_text('enough')
            return bits
        session.screen = screen
        status = session.run()
        self.assertEqual(status['outcome'], 'stopped')
        result = self.read('screen-300.json')
        self.assertEqual(sorted(result['saturated_channels']), ['55', '56'])
        self.assertFalse({55, 56} & set(self.read('decision.json')['roi']))
        self.assertLess(self.read('decision.json')['reference_hz'], self.read('warmup.json')['reference_hz'])

    def test_collapsed_background_activity_stops_the_session(self):
        status = self.session(background={'main_r1_' + k: .05 for k in ('H', 'W', 'F', 'uniform')}).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertEqual(sum(b['name'].startswith('main') for b in status['blocks']), 1)
        self.assertFalse(status['blocks'][-1]['passed'])
        self.assertEqual(status['dose'], 60 + 90)

    def test_late_start_ends_at_the_cutoff_without_late_pulses(self):
        status = self.session(age_min=70.).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertIn('time', status['stop_reason'])
        rows = self.ledger()
        self.assertTrue(rows)
        self.assertLess(max(r['age_s'] + r['seconds'] for r in rows), design.STIM_CUTOFF_S)
        main = [b for b in status['blocks'] if b['name'].startswith('main')]
        self.assertTrue(0 < len(main) < 12)
        self.assertTrue(all(b['passed'] for b in main))
        self.assertLess(status['age_s'], design.HARD_STOP_S)

    def test_gate_refuses_a_chunk_beyond_the_budget_or_cutoff(self):
        session = self.session()
        event = dict(id=1, block='x', first=0, trials=10, real=9, seconds=18.)
        session.dose = design.MAX_REAL_PULSES - 8
        with self.assertRaises(Stop):
            session.reserve(event)
        self.assertEqual(session.dose, design.MAX_REAL_PULSES - 8)
        self.assertEqual(self.ledger(), [])
        session.dose = 0
        session.reserve(event)
        self.assertEqual(self.ledger()[0]['total'], 9)
        self.clock.sleep(design.STIM_CUTOFF_S - 600 - 10)
        with self.assertRaises(Stop):
            session.reserve(event)
        session.reserve(dict(event, real=0))      # a chunk without output is not a pulse
        self.assertEqual(session.dose, 9)

    def test_hardware_fault_skips_recovery_and_goes_straight_to_cleanup(self):
        session = self.session()
        original = self.rig.send
        def send(handle, event, gate):
            if event['block'] == 'screen300_2':
                self.rig.faulted = True
                raise RuntimeError('lost acknowledgement')
            original(handle, event, gate)
        self.rig.send = send
        status = session.run()
        self.assertEqual(status['outcome'], 'error')
        self.assertIn('lost acknowledgement', status['error'])
        self.assertNotIn('recovery', status)
        self.assertTrue(self.rig.cleaned)
        self.assertEqual(status['dose'], 30)

    def test_changed_plan_is_refused(self):
        changed = dict(self.plan, seed=32)
        with self.assertRaises(ValueError):
            Session(None, self.out, changed, 0, SESSION, [], self.clock)


if __name__ == '__main__':
    unittest.main()
