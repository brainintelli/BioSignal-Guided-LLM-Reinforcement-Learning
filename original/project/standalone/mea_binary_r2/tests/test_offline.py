import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from mea_binary_r2.offline import run_offline
except ImportError:
    run_offline=None


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(run_offline,'offline runner has not been implemented')

    def test_full_pipeline_computes_signed_rewards_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            with patch.object(socket,'socket',side_effect=AssertionError('No network allowed')):
                result=run_offline(out)
            self.assertTrue(result['passed'])
            self.assertEqual(result['source'],'simulation')
            self.assertFalse(result['hardware_commands_sent'])
            self.assertFalse(result['model_training_performed'])
            self.assertEqual((result['positive_rows'],result['negative_rows'],result['terminal_rows']),(3,4,1))
            self.assertEqual((result['simulated_nonzero_events'],result['simulated_pulses']),(4,8))
            rows=[json.loads(s) for s in (out/'transitions.jsonl').read_text().splitlines()]
            self.assertEqual([r['reward']['r2_raw'] for r in rows],[1,-1,-1,1,-1,-1,1,None])
            self.assertEqual(rows[1]['intent']['site'],'W')
            self.assertEqual(rows[4]['intent']['site'],'F')
            self.assertEqual(rows[5]['intent']['site'],'H')
            self.assertTrue(all(r['reward']['valid'] for r in rows))
            rejected=[json.loads(s) for s in (out/'rejections.jsonl').read_text().splitlines()]
            self.assertEqual({r['case'] for r in rejected},
                             {'unknown_output','recovery_failure','missing_coverage','wrong_context','duplicate_window'})
            self.assertTrue(all(r['rejected'] for r in rejected))
            self.assertIsNone(next(r for r in rejected if r['case']=='unknown_output')['reward']['r2_raw'])

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';run_offline(out)
            original=(out/'result.json').read_bytes()
            with self.assertRaises(FileExistsError):run_offline(out)
            self.assertEqual((out/'result.json').read_bytes(),original)

    def test_cli_is_simulation_only_and_reports_real_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'cli-run'
            env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
            command=[sys.executable,'-m','mea_binary_r2.offline','--output',str(out)]
            completed=subprocess.run(command,env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(completed.returncode,0,completed.stderr)
            result=json.loads(completed.stdout)
            self.assertTrue(result['passed'])
            self.assertEqual(result,json.loads((out/'result.json').read_text()))
            attempt=subprocess.run(command+['--live'],env=env,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(attempt.returncode,0)
            self.assertIn('unrecognized arguments',attempt.stderr)


if __name__=='__main__':unittest.main()
