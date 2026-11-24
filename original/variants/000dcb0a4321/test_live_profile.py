import copy
import json
from pathlib import Path
import unittest
from contextlib import contextmanager
import signal
from run_guarded_probe import validate_profile, write_audit, deadline_safe_signals


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((Path(__file__).resolve().parent.parent /
            'mea_maxone_stimulation_parameters_20261117/package/BrainStimulation/profiles/maxone_first_probe.template.json').read_text())
        self.profile = copy.deepcopy(self.template)
        self.profile['electrodes_selection'] = {'recording': [2, 3], 'stimulation': [2]}
        self.profile['assay_path'] = '/tmp/current.cfg'

    def test_exact_profile_with_recorded_stim_site(self):
        validate_profile(self.profile, self.template)

    def test_amplitude_count_raw_mode_or_extra_site_rejected(self):
        for key, value in [('amplitude_range', [600, 600, 1]), ('pulse_trains', 6),
                           ('record_only_spikes', True), ('phase', 400),
                           ('control', True), ('post_stim_period', 0)]:
            with self.subTest(key=key):
                p = copy.deepcopy(self.profile)
                p[key] = value
                with self.assertRaises(ValueError):
                    validate_profile(p, self.template)
        self.profile['electrodes_selection']['stimulation'] = [4]
        with self.assertRaises(ValueError):
            validate_profile(self.profile, self.template)

    def test_failed_audit_never_blocks_cleanup_but_can_block_stimulus(self):
        class BrokenFile:
            def write(self, value):
                raise OSError('disk full')
        self.assertFalse(write_audit(BrokenFile(), {'kind':'cleanup'}))
        with self.assertRaises(OSError):
            write_audit(BrokenFile(), {'kind':'send_attempt'}, strict=True)

    def test_deadline_cancelled_before_cleanup_context(self):
        calls = []
        @contextmanager
        def fake_context(handler):
            calls.append('enter')
            yield
        guarded = deadline_safe_signals(fake_context, lambda:calls.append('cancel_alarm'))
        with guarded(signal.SIG_IGN):
            calls.append('cleanup_command')
        self.assertEqual(calls, ['cancel_alarm','enter','cleanup_command'])

if __name__ == '__main__':
    unittest.main()
