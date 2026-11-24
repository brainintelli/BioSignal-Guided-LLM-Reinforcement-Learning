import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import os
import signal
import subprocess
import sys
from tools import mea_p1_acquire as m


class FakeApi:
    def __init__(self, fail=None): self.calls = []; self.fail = fail
    def send(self, command, **kw):
        self.calls.append(command)
        if self.fail and command.startswith(self.fail): return 'Error'
        return 'Ok'


class AcquireTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'POSIX signal inheritance')
    def test_restored_child_does_not_inherit_ignored_termination(self):
        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = None
        try:
            child = m.start_legacy_process([sys.executable, '-c', 'import time; time.sleep(10)'], subprocess.DEVNULL)
            child.terminate()
            self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, previous)
            if child and child.poll() is None: child.kill(); child.wait()

    def test_actual_format_acknowledgement_is_command_specific(self):
        api = Mock(); api.send.return_value = 'Using new file format'
        client = m.CheckedApi(api)
        self.assertEqual(client.send('saving_use_old_file_format 0'), 'Using new file format')
        with self.assertRaises(RuntimeError): client.send('saving_start_file wrong_reply')

    def test_open_directory_echo_must_match_requested_path(self):
        api = Mock(); api.send.return_value = '/home/researcher/mea-acquisition/new'
        client = m.CheckedApi(api)
        self.assertEqual(client.send('saving_open_dir /home/researcher/mea-acquisition/new'), api.send.return_value)
        with self.assertRaises(RuntimeError): client.send('saving_open_dir /home/researcher/mea-acquisition/different')

    def test_file_echo_and_stop_ack_match_vendor_contract(self):
        api = Mock(); client = m.CheckedApi(api)
        api.send.return_value = '/home/researcher/owned'
        client.send('saving_open_dir /home/researcher/owned')
        api.send.return_value = '/home/researcher/owned/baseline.raw.h5'
        client.send('saving_start_file baseline')
        api.send.return_value = '/home/researcher/other/baseline.raw.h5'
        with self.assertRaises(RuntimeError): client.send('saving_start_file baseline')
        api.send.return_value = 'Stopped'; client.send('saving_stop_file')

    def test_start_recording_empty_framed_reply_is_specific_and_errors_still_raise(self):
        api = Mock(); client = m.CheckedApi(api); api.send.return_value = ''
        self.assertEqual(client.send('saving_start_recording'), '')
        with self.assertRaises(RuntimeError): client.send('saving_stop_file')
        api.send.return_value = 'Stopped Recording'; client.send('saving_stop_recording')
        api.send.side_effect = OSError('server closed connection')
        with self.assertRaises(OSError): client.send('saving_start_recording')

    def test_shutdown_exception_does_not_prevent_restoration_and_manifest(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'events.jsonl').write_text(json.dumps(dict(type='end', sdk_close_status=0))+'\n')
            api = Mock(); api.api.shutdown.side_effect = OSError('shutdown failed')
            collector = Mock(); collector.poll.return_value = 0
            restored = Mock(); restored.poll.return_value = None; restored.pid = 123
            record = dict(errors=[], success=True)
            with patch.object(m, 'alive', return_value=False), patch.object(m.time, 'sleep'), \
                 patch.object(m.subprocess, 'Popen', return_value=restored) as start:
                m.finalize_capture(None, collector, None, api, record, root, 1, {}, True)
            start.assert_called_once()
            stored = json.loads((root/'acquisition.json').read_text())
            self.assertTrue(stored['legacy_restored'])
            self.assertFalse(stored['success'])
            self.assertIn('shutdown failed', stored['errors'][0])

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
