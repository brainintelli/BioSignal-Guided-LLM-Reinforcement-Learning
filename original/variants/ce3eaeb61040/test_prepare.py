import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('prep', ROOT / 'prepare.py')


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'prepare.py').exists():
            return
        cls.m = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue((ROOT / 'prepare.py').exists(), 'offline preparation implementation missing')

    def test_microvolt_target_is_blocked_not_zero_stimulation(self):
        r = self.m.quantize('43.08894358057273', '2.981901')
        self.assertEqual(r['cap_mv'], '0.051706732296687276')
        self.assertEqual(r['offset_bits'], 0)
        self.assertFalse(r['nonzero_encodable'])
        self.assertIsNone(r['candidate_phase_mv'])

    def test_floor_and_exact_step_never_round_up(self):
        a = self.m.quantize('1000', '1.2')
        b = self.m.quantize('999.99999999999999', '1.2')
        c = self.m.quantize('1999', '1.2')
        self.assertEqual(a['offset_bits'], 1)
        self.assertEqual(b['offset_bits'], 0)
        self.assertEqual(c['offset_bits'], 1)
        self.assertEqual(c['candidate_phase_mv'], '1.2')

    def test_bipolar_dac_range_is_bounded(self):
        r = self.m.quantize('10000000', '1')
        self.assertEqual(r['offset_bits'], 511)
        self.assertEqual(r['candidate_phase_mv'], '511')

    def test_invalid_voltage_does_not_silently_pass(self):
        for x in (0, -1, True, None, 'NaN', 'Infinity', 'bad'):
            for args in ((x, '2.981901'), ('40', x)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    self.m.quantize(*args)

    def test_deadline_includes_recovery_and_closeout(self):
        self.assertTrue(self.m.budget(6479, 0.5)['fits_before_recovery'])
        self.assertFalse(self.m.budget(6479, 1)['fits_before_recovery'])
        self.assertFalse(self.m.budget(6480)['fits_before_recovery'])
        self.assertFalse(self.m.budget(7200)['fits_before_recovery'])
        self.assertEqual(self.m.budget(7200)['remaining_seconds'], 0)

    def test_invalid_time_and_future_start_rejected(self):
        for value in (-1, True, None, 'NaN', 'Infinity'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.m.budget(value)

    def fixture(self):
        return ({'mode': 'preparation_only', 'execution_enabled': False, 'r2_enabled': False,
                 'session': {'device_host': None, 'chip_id': None, 'organoid_id': None,
                             'started_at': None}, 'reference_chip_id': '22554'},
                {'schema': 'mea.p90_peak_cap.feasibility.v1', 'limits': [
                    {'chip_id': '22554', 'scope': 'pooled_reference', 'p90_uv': '10000',
                     'electrode_id': None, 'device_reported_lsb_mv': '1'},
                    {'chip_id': '22554', 'scope': 'per_electrode', 'p90_uv': '40',
                     'electrode_id': 9764, 'device_reported_lsb_mv': '2.981901'}]})

    def test_pooled_reference_never_substitutes_for_electrode(self):
        s, data = self.fixture()
        r = self.m.prepare(s, data, 'abc')
        self.assertEqual(len(r['electrodes']), 1)
        self.assertEqual(r['electrodes'][0]['electrode_id'], 9764)
        self.assertFalse(r['execution_enabled'])
        self.assertIn('historical_reference_not_current_calibration', r['blockers'])
        self.assertIn('missing_device_host', r['blockers'])

    def test_identity_and_hypothetical_encodability_never_arm(self):
        s, data = self.fixture()
        s['session'] = {'device_host': 'test', 'chip_id': 'new', 'organoid_id': 'new',
                        'started_at': '2026-11-17T00:00:00+08:00'}
        data['limits'][1]['p90_uv'] = '10000'
        r = self.m.prepare(s, data, 'abc')
        self.assertTrue(r['electrodes'][0]['nonzero_encodable'])
        self.assertFalse(r['execution_enabled'])
        self.assertIsNone(r['vendor_configuration'])
        self.assertIn('historical_reference_not_current_calibration', r['blockers'])

    def test_no_candidate_unknown_chip_or_duplicate_is_error(self):
        for alter in ('unknown', 'duplicate', 'pooled_only'):
            s, data = self.fixture()
            if alter == 'unknown':
                s['reference_chip_id'] = 'missing'
            elif alter == 'duplicate':
                data['limits'].append(copy.deepcopy(data['limits'][1]))
            else:
                data['limits'].pop()
            with self.subTest(alter=alter), self.assertRaises(ValueError):
                self.m.prepare(s, data, 'abc')

    def test_unsafe_mode_or_timestamp_not_accepted(self):
        for field, value in (('execution_enabled', True), ('r2_enabled', True),
                             ('mode', 'run')):
            s, data = self.fixture()
            s[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.m.prepare(s, data, 'abc')
        s, data = self.fixture()
        s['session']['started_at'] = '2026-11-17T00:00:00'
        with self.assertRaises(ValueError):
            self.m.prepare(s, data, 'abc')

    def test_future_session_start_rejected(self):
        s, data = self.fixture()
        s['session']['started_at'] = '2026-11-17T00:00:01+00:00'
        with self.assertRaises(ValueError):
            self.m.prepare(s, data, 'abc', now=datetime(2026, 11, 17, tzinfo=timezone.utc))

    def test_session_at_deadline_is_blocked_in_report(self):
        s, data = self.fixture()
        s['session']['started_at'] = '2026-11-17T00:00:00+00:00'
        r = self.m.prepare(s, data, 'abc', now=datetime(2026, 11, 17, 2, tzinfo=timezone.utc))
        self.assertEqual(r['timing']['remaining_seconds'], 0)
        self.assertIn('time_budget_does_not_allow_new_output', r['blockers'])
        self.assertFalse(r['hardware_deadline_enforced'])

    def test_cli_writes_blocked_report_and_refuses_overwrite(self):
        s, data = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 's.json').write_text(json.dumps(s), encoding='utf-8')
            (p / 'r.json').write_text(json.dumps(data), encoding='utf-8')
            cmd = [sys.executable, str(ROOT / 'prepare.py'), '--session', str(p/'s.json'),
                   '--reference', str(p/'r.json'), '--output', str(p/'out.json')]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 2, first.stderr)
            before = (p/'out.json').read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 1)
            self.assertEqual((p/'out.json').read_bytes(), before)
            self.assertFalse(json.loads(before)['hardware_connected'])

    def test_cli_numeric_decimal_token_cannot_round_up_to_one_step(self):
        s, data = self.fixture()
        data['limits'][1]['p90_uv'] = 'EXACT_TOKEN'
        data['limits'][1]['device_reported_lsb_mv'] = 'EXACT_LSB'
        raw = json.dumps(data).replace('"EXACT_TOKEN"', '999.99999999999999').replace('"EXACT_LSB"', '1.2')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'s.json').write_text(json.dumps(s), encoding='utf-8')
            (p/'r.json').write_text(raw, encoding='utf-8')
            result = subprocess.run([sys.executable, str(ROOT/'prepare.py'), '--session',
                                     str(p/'s.json'), '--reference', str(p/'r.json'),
                                     '--output', str(p/'out.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            row = json.loads((p/'out.json').read_text(encoding='utf-8'))['electrodes'][0]
            self.assertEqual(row['offset_bits'], 0)
            self.assertEqual(row['cap_mv'], '1.199999999999999988')


if __name__ == '__main__':
    unittest.main()
