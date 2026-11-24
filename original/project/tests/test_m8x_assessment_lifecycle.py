"""CPU lifecycle fixtures: no GPU, model, world or remote process is started."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from tools import m8x_remote_assessment as assessment


class AssessmentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'output/runtime_tests')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / 'm8b-evidence'
        self.evidence.mkdir()
        self.run = self.root / 'm8b-runs/run'
        self.run.mkdir(parents=True)
        self.request = self.evidence / 'request.json'
        self.data = dict(run=str(self.run), output=str(self.run.parent / 'test'), mode='test')
        self.request.write_text(json.dumps(self.data), encoding='utf-8')
        self.jobs = self.evidence / 'assessments'
        self.overlay=self.root/'m8b-runtime-candidates/test/site';self.overlay.mkdir(parents=True)
        self.project=self.root/'m8b-project';self.project.mkdir()
        self.config=dict(kernel_overlay=str(self.overlay),project=str(self.project))

    def start(self, kind='exploration-stochastic', gpu='', rejected=False):
        argv = ['assessment', 'start', '--name', 'fixture', '--kind', kind, '--request', str(self.request)]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(assessment, 'ROOT', self.root))
            stack.enter_context(patch.object(assessment, 'JOBS', self.jobs))
            stack.enter_context(patch.object(assessment.subprocess, 'check_output', return_value=gpu))
            stack.enter_context(patch.object(assessment.subprocess, 'Popen', return_value=MagicMock(pid=123)))
            stack.enter_context(patch.object(assessment.lifecycle, 'identity', return_value=dict(pid=123, start_ticks='1', executable='/python')))
            stack.enter_context(patch.object(assessment.time, 'sleep'))
            stack.enter_context(patch.object(assessment, 'authorize_test', side_effect=ValueError('Selection not accepted') if rejected else None,return_value=(self.config,None,None,None,None,None)))
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            assessment.main()

    def test_routes_to_frozen_evaluator_with_isolated_runtime(self):
        self.start()
        job = self.jobs / 'fixture'
        launch = json.loads((job / 'launch.json').read_text(encoding='utf-8'))
        self.assertEqual(launch['argv'][3], 'agent.m8x_eval')
        self.assertEqual((job / 'request.json').read_bytes(), self.request.read_bytes())
        self.assertEqual(launch['hard_wall_seconds'], 172800)
        self.assertEqual(launch['pythonpath'],str(self.overlay)+':'+str(self.project))

    def test_missing_independent_acceptance_creates_no_job(self):
        with self.assertRaisesRegex(ValueError,'Selection not accepted'):
            self.start(rejected=True)
        self.assertFalse(self.jobs.exists())

    def test_validation_mode_is_rejected_before_launch(self):
        self.request.write_text(json.dumps({**self.data, 'mode': 'validation'}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Exploration.*test'):
            self.start()
        self.assertFalse(self.jobs.exists())

    def test_busy_gpu_is_untouched_and_no_job_created(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU busy'):
            self.start(gpu='2344431, training, 45270 MiB')
        self.assertFalse(self.jobs.exists())

    def test_existing_output_is_not_overwritten(self):
        Path(self.data['output']).mkdir()
        with self.assertRaisesRegex(ValueError, 'Evidence path already exists'):
            self.start()
        self.assertFalse(self.jobs.exists())


if __name__ == '__main__':
    unittest.main()
