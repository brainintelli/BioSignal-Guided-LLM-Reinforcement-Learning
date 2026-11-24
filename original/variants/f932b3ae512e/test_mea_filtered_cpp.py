import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name=='posix' and shutil.which('g++'),'Linux SDK 1.1 collector')
class FilteredCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.binary=cls.root/'capture'
        subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-DMEA_FILTERED_FRAME_INFO',
                        '-I'+str(ROOT/'tests/mea_sdk_stub'),str(ROOT/'tools/mea_timing_capture.cpp'),
                        '-o',str(cls.binary)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_live_progress_is_ordered_after_events_and_covers_quiet_frames(self):
        from agent.mea_timing import FilteredFrameStream
        path=self.root/'good.jsonl'
        p=subprocess.run([str(self.binary),str(path),'1'],env={**os.environ,'MEA_STUB':'filtered_progress'},timeout=5)
        self.assertEqual(p.returncode,0)
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        self.assertEqual(rows[0]['protocol'],'mea.filtered.transport.v1')
        stream=FilteredFrameStream();stream.feed(path.read_bytes());stream.finish()
        self.assertEqual(stream.watermark,102500)
        self.assertEqual(stream.frame_count,2500)
        self.assertEqual(stream.quiet_blocks,2)

    def test_filtered_gaps_corruption_and_event_frame_mismatch_fail(self):
        for mode in ('filtered_gap','filtered_corrupt','event_frame_mismatch'):
            p=subprocess.run([str(self.binary),str(self.root/(mode+'.jsonl')),'1'],env={**os.environ,'MEA_STUB':mode},timeout=5)
            self.assertNotEqual(p.returncode,0)

    def test_delayed_peak_preserves_both_frame_counters(self):
        from agent.mea_timing import FilteredFrameStream
        path=self.root/'delayed.jsonl'
        p=subprocess.run([str(self.binary),str(path),'1'],env={**os.environ,'MEA_STUB':'delayed_peak'},timeout=5)
        self.assertEqual(p.returncode,0)
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        event=next(r for r in rows if r['type']=='events')
        self.assertEqual(event['frame_number']-event['events'][0][0],1880)
        stream=FilteredFrameStream();stream.feed(path.read_bytes());stream.finish()
        self.assertEqual(stream.watermark,102500)
