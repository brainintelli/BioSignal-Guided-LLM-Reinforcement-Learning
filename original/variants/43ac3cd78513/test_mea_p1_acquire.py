import tempfile
import unittest
from pathlib import Path
from tools import mea_p1_acquire as m


class FakeApi:
    def __init__(self, fail=None): self.calls = []; self.fail = fail
    def send(self, command, **kw):
        self.calls.append(command)
        if self.fail and command.startswith(self.fail): return 'Error'
        return 'Ok'


class AcquireTests(unittest.TestCase):
    def test_mapping_is_complete_unique_and_finite(self):
        self.assertEqual(m.parse_mapping('0(100)1/2;1(101)3.5/-4;'), [[0, 100], [1, 101]])
        for bad in ('0(100)1/2', '0(100)1/2;0(101)3/4;', '0(100)nan/2;', '',
                    '0(100)1/2;1(100)3/4;', '0(100)1/2;junk;'):
            with self.subTest(bad=bad), self.assertRaises(ValueError): m.parse_mapping(bad)

    def test_reject_api_error_and_forbidden_commands(self):
        api = FakeApi('saving_start_file'); client = m.CheckedApi(api)
        with self.assertRaises(RuntimeError): client.send('saving_start_file owned')
        self.assertEqual(len(client.log), 1)
        for command in ('mea_set_stimulation', 'saving_group_clear', 'mea_array_delete online',
                        'saving_start_file ok\nmea_set_stimulation'):
            with self.assertRaises(ValueError): client.send(command)
        self.assertEqual(len(api.calls), 1)

    def test_cleanup_only_owned_recording_and_closes_file_on_failure(self):
        api = FakeApi(); owner = m.RecordingOwner(m.CheckedApi(api))
        owner.cleanup(); self.assertEqual(api.calls, [])
        owner.file_started = True; owner.recording_started = True; owner.group_name = 'p1_raw'
        owner.cleanup()
        self.assertEqual(api.calls, ['saving_stop_recording', 'saving_stop_file', 'saving_group_delete 0 p1_raw'])
        owner.cleanup(); self.assertEqual(len(api.calls), 3)

    def test_cleanup_attempts_file_close_after_record_stop_error(self):
        api = FakeApi('saving_stop_recording'); owner = m.RecordingOwner(m.CheckedApi(api))
        owner.file_started = True; owner.recording_started = True
        with self.assertRaises(RuntimeError): owner.cleanup()
        self.assertIn('saving_stop_file', api.calls)

    def test_exclusive_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'new'; m.exclusive_directory(path)
            with self.assertRaises(FileExistsError): m.exclusive_directory(path)
