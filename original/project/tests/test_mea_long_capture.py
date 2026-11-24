import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace


class LongCaptureTests(unittest.TestCase):
    def test_explicit_long_duration_and_uncompressed_disk_budget(self):
        m=importlib.import_module('tools.mea_p1_acquire')
        budget=m.capture_budget(1800)
        self.assertEqual(budget['collector_seconds'],1820)
        self.assertGreaterEqual(budget['required_free_bytes'],1800*20000*1024*2+5_000_000_000)
        self.assertEqual(m.capture_budget(60)['collector_seconds'],80)
        for invalid in [4,1801,True,1800.5]:
            with self.assertRaises(ValueError):m.capture_budget(invalid)

    def test_low_space_rejects_and_progress_is_explicitly_not_complete(self):
        m=importlib.import_module('tools.mea_p1_acquire')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=1_999_999_999)):
                with self.assertRaises(RuntimeError):m.write_capture_progress(p,1800,10.)
            with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=50_000_000_000)):
                m.write_capture_progress(p,1800,10.)
            import json
            state=json.loads((p/'acquisition_progress.json').read_text())
            self.assertEqual(state['requested_seconds'],1800)
            self.assertEqual(state['elapsed_seconds'],10.)
            self.assertFalse(state['peak_windows_complete'])
            self.assertFalse(state['realtime_training'])
            self.assertFalse(state['stimulation'])
