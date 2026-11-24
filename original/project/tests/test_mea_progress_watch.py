import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from tools import mea_progress_watch as w


@unittest.skipUnless(os.name=='posix','source monotonic raw clock')
class WatchTests(unittest.TestCase):
    def test_live_gate_and_timeout(self):
        from tests.test_mea_filtered_progress import packets
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=packets()
            for row in rows:
                if row['type']=='frame_progress':row['recv_end_ns']=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            path=root/'events';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report=w.watch(path,root/'seen.jsonl',1)
            self.assertTrue(report['passed']);self.assertEqual(report['final_watermark'],300)
            report=w.watch(root/'absent',root/'timeout.jsonl',.03)
            self.assertFalse(report['passed'])
