import sys
from pathlib import Path
from contextlib import contextmanager
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mea_maxone_stimulation_parameters_20261117/package/BrainStimulation'))
import capture_qualification as capture


class SavingAllowlistTest(unittest.TestCase):
    def test_full_sdk_saving_chain_has_no_unrecognized_command(self):
        messages = []
        class Fake:
            def send(self, message, **kwargs):
                messages.append(message)
                return 'OK'
        @contextmanager
        def fake_context(*args, **kwargs):
            yield Fake()
        old = capture.mx.comm.api_context
        capture.mx.comm.api_context = fake_context
        try:
            s = capture.mx.Saving()
            s.open_directory(str(capture.ROOT))
            s.set_legacy_format(False)
            s.start_file('qualification')
            s.group_delete_all()
            s.group_define(0, 'routed', list(range(1024)))
            s.start_recording([0])
            s.stop_recording()
            s.stop_file()
            s.group_delete_all()
        finally:
            capture.mx.comm.api_context = old
        self.assertTrue(messages)
        self.assertEqual(set(messages) - capture.ALLOWED, set())
        self.assertFalse(any('sequence' in c or 'dac' in c or 'initialize' in c for c in messages))

if __name__ == '__main__':
    unittest.main()
