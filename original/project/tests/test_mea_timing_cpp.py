import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from agent.mea_recording import load_live

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'Linux SDK collector')
class TimingCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.binary = cls.root/'capture'
        subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',
                        '-I'+str(ROOT/'tests/mea_sdk_stub'),str(ROOT/'tools/mea_timing_capture.cpp'),
                        '-o',str(cls.binary)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def run_mode(self, mode):
        output = self.root/(mode+'.jsonl')
        result = subprocess.run([str(self.binary),str(output),'1'],env={**os.environ,'MEA_STUB':mode},timeout=5)
        rows = [json.loads(s) for s in Path(str(output)+'.progress.jsonl').read_text().splitlines()]
        return result, output, rows

    def test_raw_quiet_frames_and_partial_block_advance_independently(self):
        result,path,rows = self.run_mode('quiet')
        self.assertEqual(result.returncode,0)
        blocks = [r for r in rows if r['type']=='raw_block']
        self.assertEqual([r['count'] for r in blocks],[1000,1000,500])
        self.assertEqual(blocks[-1]['end_frame'],102500)
        self.assertEqual(rows[-1]['raw_frames'],2500)
        self.assertEqual([r['seq'] for r in rows],list(range(len(rows))))
        events,status = load_live(path); self.assertTrue(status['clean_end'])
        self.assertTrue(rows[-1]['raw_contiguous'])
        saved = path.read_bytes()
        self.assertNotEqual(subprocess.run([str(self.binary),str(path),'1'],timeout=5).returncode,0)
        self.assertEqual(path.read_bytes(),saved)

    def test_raw_faults_and_sdk_failures_cannot_pass(self):
        for mode in ('raw_gap','raw_reset','corrupt','raw_open_fail','raw_close_fail','close_fail'):
            with self.subTest(mode=mode):
                result,path,rows = self.run_mode(mode)
                self.assertNotEqual(result.returncode,0)
                events,status = load_live(path)
                self.assertFalse(status['clean_end'])
                self.assertFalse(rows[-1]['clean_end'])

    def test_startup_wait_is_not_reported_as_zero_age(self):
        result,path,rows=self.run_mode('startup_stall')
        self.assertEqual(result.returncode,0)
        first=next(r for r in rows if r['type']=='heartbeat')
        self.assertGreaterEqual(first['since_raw_ns'],250_000_000)
