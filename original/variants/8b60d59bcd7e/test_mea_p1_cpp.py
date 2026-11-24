"""Collector integration tests against a fake SDK: never open real hardware."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from agent.mea_recording import load_live

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'Linux collector build test')
class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.binary = cls.root/'collector'
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                        '-I'+str(ROOT/'tests/mea_sdk_stub'), str(ROOT/'tools/mea_p1_capture.cpp'),
                        '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_success_byte_log_and_exclusive_output(self):
        path = self.root/'ok.jsonl'
        result = subprocess.run([str(self.binary), str(path), '1'], timeout=5)
        self.assertEqual(result.returncode, 0)
        events, status = load_live(path)
        self.assertEqual(events, [[123456, 42, 0, -3.125]])
        self.assertTrue(status['clean_end'])
        original = path.read_bytes()
        result = subprocess.run([str(self.binary), str(path), '1'], timeout=5)
        self.assertNotEqual(result.returncode, 0); self.assertEqual(path.read_bytes(), original)

    def test_sdk_failures_never_become_clean_eof(self):
        for mode in ('open_fail', 'receive_fail', 'close_fail'):
            path = self.root/(mode+'.jsonl')
            result = subprocess.run([str(self.binary), str(path), '1'],
                                    env={**os.environ, 'MEA_STUB': mode}, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            events, status = load_live(path)
            self.assertFalse(status['clean_end'])
