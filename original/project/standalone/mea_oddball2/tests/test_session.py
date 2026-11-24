"""Whole sessions against a simulated rig that writes real-layout H5 files."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_library
import quicklook
from oddball2 import design, library
from oddball2.session import Session, Stop, write_once
from support import READOUT, SESSION, SITE_CHANNELS, Clock, FakeRig


class SessionCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'oddball2-test'
        self.out.mkdir()
        self.clock = Clock()
        self.plan = design.plan(31, 600)
        self.lines = []

    def session(self, age_min=10., plan=None, **rig):
        self.rig = FakeRig(self.out, self.clock, **rig)
        self.plan = plan or self.plan
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

    def stop_after(self, session, name):
        """Operator STOP file once the block `name` has closed."""
        original = session.block
        def block(item, bits):
            result = original(item, bits)
            if item['name'] == name:
                (self.out / 'STOP').write_text('operator')
            return result
        session.block = block


class SessionTests(SessionCase):
    def test_complete_session_at_the_lowest_amplitude(self):
        session = self.session()
        status = session.run()
        self.assertEqual(status['outcome'], 'complete', status.get('error'))
        self.assertEqual(status['dose'], 60 + 90 + 1080)
        decision = self.read('decision.json')
        self.assertEqual((decision['amplitude_mv'], decision['bits']), (300, 100))
        self.assertEqual(decision['confirmed_sites'], list(design.SITES))
        self.assertEqual(decision['readable'], list(design.SITES))
        self.assertTrue(self.read('screen-300.json')['confirm'])
        self.assertTrue(self.read('confirm-300.json')['proceed'])
        self.assertFalse((self.out / 'screen-450.json').exists())
        main = [b['name'] for b in self.plan['blocks'] if b['stage'] == 'main']
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300', 'confirm300'] + main + ['recovery'])
        self.assertTrue(all(b['passed'] for b in status['blocks']))
        self.assertTrue(self.rig.cleaned)
        self.assertEqual(status['cleanup'], [])
        # The amplitude never changes after the decision.
        self.assertEqual({self.read(name + '-sent.json')['bits'] for name in main}, {100})
        self.assertEqual(self.read('rehearsal-sent.json')['bits'], 0)
        # No real pulse after minute 95 and the whole session inside two hours.
        rows = self.ledger()
        self.assertEqual(rows[-1]['total'], 1230)
        self.assertLess(max(r['age_s'] for r in rows), design.STIM_CUTOFF_S)
        self.assertLess(status['age_s'], design.HARD_STOP_S)
        self.assertEqual(self.read('recovery.json')['events'], 0)
        self.assertGreaterEqual(self.read('recovery.json')['seconds'], 599)
        # One minute without recording or output between the screen and its confirmation.
        screen, confirm = self.read('screen300-sent.json')['chunks'], self.read('confirm300-sent.json')['chunks']
        self.assertGreater(confirm[0]['wall'] - screen[-1]['wall'], 60 + 20 + 10)
        # Units are connected for every stimulation file, including sham trials, and off at the end.
        self.assertFalse(self.rig.on)
        self.assertEqual([on for name, on in self.rig.unit_log if name == 'main_r1_uniform'], [True, False])
        # Chunks leave the host every 20 s inside a block.
        sent = self.read('main_r1_uniform-sent.json')['chunks']
        self.assertEqual(len(sent), 10)
        gaps = [b['wall'] - a['wall'] for a, b in zip(sent, sent[1:])]
        self.assertTrue(all(abs(g - 20) < .05 for g in gaps))
        session.manifest()
        self.assertEqual(len(self.read('manifest.json')), 1 + 1 + 1 + 1 + 12 + 1)
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            quicklook.show(self.out)
        self.assertIn('screen 300 mV (actual 298.19): run confirmation True', shown.getvalue())
        self.assertIn('confirm 300 mV (actual 298.19): run main blocks True', shown.getvalue())
        self.assertIn('site pulses booked: 1230 of 1400', shown.getvalue())
        self.assertIn('ended: complete', shown.getvalue())
        self.assertEqual(sum(line.startswith('main_r') for line in shown.getvalue().splitlines()), 12)
        # Device-detected counts give a preview; a preview never writes a library.
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            report = build_library.main(self.out, self.out / 'preview', rho_min=.2, preview=True)
        self.assertTrue(report['gates']['library_allowed'])
        self.assertEqual(report['gates']['usable_sites'], list(design.SITES))
        self.assertFalse((self.out / 'preview' / 'library.json').exists())
        self.assertIn('library written: False (preview)', printed.getvalue())
        self.assertIn('training allowed: False', printed.getvalue())

    def test_silent_sample_climbs_the_whole_ladder_and_stops_without_main_blocks(self):
        status = self.session(gain=0.).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertIn('no amplitude up to 600 mV', status['stop_reason'])
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300', 'screen450', 'screen600', 'recovery'])
        self.assertEqual([self.read('screen-%d.json' % a)['bits'] for a in (300, 450, 600)], [100, 150, 201])
        self.assertFalse(any(self.read('screen-%d.json' % a)['confirm'] for a in (300, 450, 600)))
        self.assertEqual(status['dose'], 180)
        self.assertFalse((self.out / 'decision.json').exists())
        self.assertIn('recorded', status['recovery'])
        self.assertTrue(self.rig.cleaned)
        with self.assertRaises(SystemExit):
            build_library.main(self.out, self.out / 'analysis')

    def test_the_ceiling_is_never_exceeded(self):
        status = self.session(plan=design.plan(31, 450), gain=0.).run()
        self.assertIn('no amplitude up to 450 mV', status['stop_reason'])
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300', 'screen450', 'recovery'])
        self.assertEqual(max(self.read(n + '-sent.json')['bits'] for n in ('screen300', 'screen450')), 150)

    def test_response_only_at_a_higher_amplitude_is_used_from_there(self):
        session = self.session(gain=lambda name, site: 0. if '300' in name else 60.)
        self.stop_after(session, next(b['name'] for b in self.plan['blocks'] if b['stage'] == 'main'))
        status = session.run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertFalse(self.read('screen-300.json')['confirm'])
        self.assertFalse((self.out / 'confirm-300.json').exists())
        decision = self.read('decision.json')
        self.assertEqual((decision['amplitude_mv'], decision['bits']), (450, 150))
        self.assertAlmostEqual(decision['actual_mv'], 447.28515)
        self.assertEqual(self.rig.files[:5], ['warmup', 'rehearsal', 'screen300', 'screen450', 'confirm450'])
        self.assertEqual(status['dose'], 60 + 60 + 90 + 90)

    def test_a_screen_that_is_not_confirmed_ends_stimulation_without_trying_more_amplitude(self):
        status = self.session(gain=lambda name, site: 0. if name.startswith('confirm') else 60.).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertIn('not confirmed', status['stop_reason'])
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300', 'confirm300', 'recovery'])
        self.assertFalse(self.read('confirm-300.json')['proceed'])
        self.assertFalse((self.out / 'decision.json').exists())
        self.assertEqual(status['dose'], 60 + 90)

    def test_one_responding_site_is_not_enough(self):
        status = self.session(gain=lambda name, site: 60. if site == 'H' else 0.).run()
        self.assertIn('no amplitude', status['stop_reason'])
        self.assertEqual(self.read('screen-600.json')['gate']['passed_sites'], ['H'])

    def test_two_responding_sites_carry_the_session(self):
        session = self.session(gain=lambda name, site: 0. if site == 'W' else 60.)
        self.stop_after(session, 'confirm300')
        status = session.run()
        self.assertIn('STOP', status['stop_reason'])
        self.assertEqual(self.read('decision.json')['confirmed_sites'], ['H', 'F'])

    def test_output_during_the_zero_amplitude_rehearsal_stops_before_any_pulse(self):
        def leak(name, pulses, patches, events):
            if name == 'rehearsal':
                patches.append((SITE_CHANNELS['H'][2], events[3][0], events[3][0] + 800, 1023))
        status = self.session(tamper=leak).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertEqual(status['dose'], 0)
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal'])
        self.assertNotIn('recovery', status)
        self.assertTrue(self.rig.cleaned)

    def test_missing_onsets_stop_stimulation_and_still_record_recovery(self):
        def lose(name, pulses, patches, events):
            if name == 'screen300':
                del patches[:12]
        status = self.session(tamper=lose).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertIn('screen300', status['stop_reason'])
        self.assertEqual(status['dose'], 60)
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300', 'recovery'])
        self.assertFalse((self.out / 'screen-300.json').exists())

    def test_saturated_readout_channels_leave_only_the_site_that_saturates_them(self):
        hit = READOUT['H'][:2]
        def saturate(name, pulses, patches, events):
            if name.startswith('screen'):
                patches.extend((c, p['frame'] + 250, p['frame'] + 450, 0) for p in pulses if p['site'] == 'H' for c in hit)
        session = self.session(tamper=saturate)
        self.stop_after(session, 'confirm300')
        status = session.run()
        self.assertEqual(status['outcome'], 'stopped')
        result = self.read('screen-300.json')
        self.assertEqual(sorted(result['saturated_channels']['H']), sorted(str(c) for c in hit))
        self.assertEqual(result['saturated_channels']['W'], {})
        decision = self.read('decision.json')
        self.assertFalse(set(hit) & set(decision['roi']['sites']['H']))
        self.assertEqual(len(decision['roi']['sites']['H']), 22)
        self.assertEqual(len(decision['roi']['sites']['W']), 24)
        self.assertLess(decision['reference_hz'], self.read('warmup.json')['reference_hz'])

    def test_a_site_whose_readout_saturates_is_unreadable_and_the_other_two_carry_on(self):
        hit = READOUT['H'][:6]
        def saturate(name, pulses, patches, events):
            if name.startswith('screen'):
                patches.extend((c, p['frame'] + 250, p['frame'] + 450, 0) for p in pulses if p['site'] == 'H' for c in hit)
        session = self.session(tamper=saturate)
        self.stop_after(session, 'confirm300')
        session.run()
        self.assertEqual(self.read('screen-300.json')['readable'], ['W', 'F'])
        self.assertEqual(self.read('decision.json')['confirmed_sites'], ['W', 'F'])

    def test_saturation_of_two_sites_ends_stimulation(self):
        def saturate(name, pulses, patches, events):
            if name.startswith('screen'):
                patches.extend((c, p['frame'] + 250, p['frame'] + 450, 0) for p in pulses if p['site'] in 'HW'
                               for c in READOUT[p['site']][:6])
        status = self.session(tamper=saturate).run()
        self.assertIn('too few readout channels', status['stop_reason'])
        self.assertEqual(status['dose'], 60)
        self.assertEqual(self.rig.files, ['warmup', 'rehearsal', 'screen300', 'recovery'])

    def test_collapsed_background_activity_stops_the_session(self):
        first = next(b['name'] for b in self.plan['blocks'] if b['stage'] == 'main')
        status = self.session(background={first: .05}).run()
        self.assertEqual(status['outcome'], 'stopped')
        self.assertEqual(sum(b['name'].startswith('main') for b in status['blocks']), 1)
        self.assertFalse(status['blocks'][-1]['passed'])
        self.assertEqual(status['dose'], 60 + 90 + 90)

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

    def test_latest_permitted_start_still_completes_two_rounds_after_a_full_ladder(self):
        from oddball2.session import START_LIMIT_S
        gain = lambda name, site: 60. if name in ('screen600', 'confirm600') or name.startswith('main') else 0.
        status = self.session(age_min=START_LIMIT_S / 60, gain=gain).run()
        done = [b['name'] for b in status['blocks'] if b['name'].startswith('main') and b['passed']]
        self.assertGreaterEqual(len(done), 8)
        self.assertEqual(self.read('decision.json')['amplitude_mv'], 600)
        self.assertLessEqual(status['dose'], design.MAX_REAL_PULSES)

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
            if event['block'] == 'confirm300':
                self.rig.faulted = True
                raise RuntimeError('lost acknowledgement')
            original(handle, event, gate)
        self.rig.send = send
        status = session.run()
        self.assertEqual(status['outcome'], 'error')
        self.assertIn('lost acknowledgement', status['error'])
        self.assertNotIn('recovery', status)
        self.assertTrue(self.rig.cleaned)
        self.assertEqual(status['dose'], 60)

    def test_changed_plan_is_refused(self):
        changed = dict(self.plan, seed=32)
        with self.assertRaises(ValueError):
            Session(None, self.out, changed, 0, SESSION, [], self.clock)


class RawLibraryTests(SessionCase):
    """The registered path: counts from raw samples, gates on the held-out round, library of usable sites."""

    def test_two_rounds_recounted_from_raw_samples_build_a_library_of_the_responding_sites(self):
        main = [b['name'] for b in self.plan['blocks'] if b['stage'] == 'main']
        session = self.session(raw=True, gain=lambda name, site: 0. if site == 'W' and name.startswith('main') else 40.)
        self.stop_after(session, main[7])
        status = session.run()
        self.assertEqual(status['outcome'], 'stopped')
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            report = build_library.main(self.out, self.out / 'analysis')
        gates = report['gates']
        self.assertEqual(sum(b['passed'] for b in report['blocks']), 8)
        self.assertEqual((gates['library_rounds'], gates['held_out_round']), ([1], 2))
        self.assertEqual(gates['usable_sites'], ['H', 'F'])
        self.assertTrue(gates['library_allowed'])
        self.assertFalse(gates['training_allowed'])                  # no stage 0 threshold supplied
        self.assertGreater(gates['reliability']['rho'], .2)
        self.assertIn('library written: True', printed.getvalue())
        self.assertIn('training allowed: False (reliability threshold from stage 0 not supplied yet)', printed.getvalue())
        made = library.check(json.loads((self.out / 'analysis' / 'library.json').read_text()))
        self.assertEqual(sorted(made['sites']), ['F', 'H'])
        self.assertEqual(report['library_sha256'], made['sha256'])
        for site in ('H', 'F'):
            means = made['stats'][site]['bin_means']
            self.assertGreater(means['low'], means['high'])
        # Raw counts follow the device's counts closely where there is no artifact to remove.
        with (self.out / 'analysis' / 'trials.csv').open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 800)
        raw = [float(r['r_H']) for r in rows]
        device = [float(r['device_r_H']) for r in rows]
        self.assertGreater(np.corrcoef(raw, device)[0, 1], .95)
        psth = json.loads((self.out / 'analysis' / 'psth.json').read_text())
        after = [i for i, t in enumerate(psth['bin_start_ms']) if 5 <= t < 50]
        real, sham = psth['sites']['H']['real']['spikes_per_trial'], psth['sites']['H']['sham']['spikes_per_trial']
        self.assertGreater(sum(real[i] for i in after), sum(sham[i] for i in after) + 5)
        thresholds = json.loads((self.out / 'analysis' / 'thresholds.json').read_text())
        self.assertEqual(len(thresholds['channels']), 72)
        with self.assertRaises(FileExistsError):
            build_library.main(self.out, self.out / 'analysis')
        # With the stage 0 threshold the same data clears or fails training, and nothing else changes.
        with contextlib.redirect_stdout(io.StringIO()):
            cleared = build_library.main(self.out, self.out / 'analysis-rho', rho_min=.2)
        self.assertTrue(cleared['gates']['training_allowed'])
        self.assertEqual(cleared['gates']['usable_sites'], gates['usable_sites'])
        self.assertEqual(cleared['gates']['reliability']['rho'], gates['reliability']['rho'])
        stored = json.loads((self.out / 'analysis-rho' / 'library.json').read_text())
        self.assertEqual(stored['sites'], made['sites'])
        self.assertEqual((stored['reliability']['rho_min'], stored['reliability']['passed']), (.2, True))
        self.assertIsNone(made['reliability']['passed'])


if __name__ == '__main__':
    unittest.main()
