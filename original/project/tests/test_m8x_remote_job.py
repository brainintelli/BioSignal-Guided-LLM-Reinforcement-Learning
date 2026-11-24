"""Lifecycle authorization and argv checks; no real process is launched."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ExplorationRemoteJobTests(unittest.TestCase):
    def test_owned_launch_uses_new_trainer_and_preserves_gpu_work(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.m8x_remote_job'))
        m=importlib.import_module('tools.m8x_remote_job')
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace/'output/runtime_tests') as tmp:
            root=Path(tmp)
            for folder in ('m8b-project','m8b-evidence','models/base','godot'):(root/folder).mkdir(parents=True)
            godot=root/'godot/engine';godot.write_text('fixture')
            overlay=root/'m8b-runtime-candidates/test/site';overlay.mkdir(parents=True)
            config=root/'m8b-evidence/config.json';config.write_text(json.dumps(dict(
                project=str(root/'m8b-project'),output=str(root/'m8b-runs/new'),model=str(root/'models/base'),godot=str(godot),kernel_overlay=str(overlay))))
            args=['job','start','--name','fixture','--config',str(config)]
            identity=dict(pid=123,start_ticks='999',executable='synthetic')
            with patch.object(m,'ROOT',root),patch.object(m,'JOBS',root/'m8b-evidence/jobs'),patch('sys.argv',args),\
                 patch.object(m,'validate_config') as validate,patch.object(m.subprocess,'check_output',return_value='987, other, 1024 MiB') as gpu,\
                 patch.object(m.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as popen,\
                 patch.object(m,'identity',return_value=identity),patch.object(m.time,'sleep'):
                with self.assertRaisesRegex(RuntimeError,'GPU has compute'):m.main()
                popen.assert_not_called();self.assertFalse((root/'m8b-evidence/jobs/fixture').exists())
                gpu.return_value='';m.main();validate.assert_called()
                launch=json.loads((root/'m8b-evidence/jobs/fixture/launch.json').read_text())
                self.assertEqual(launch['argv'][3],'agent.m8x_train')
                self.assertEqual(launch['hard_wall_seconds'],172800)
                self.assertEqual(launch['pythonpath'],str(overlay)+':'+str(root/'m8b-project'))
                self.assertEqual((root/'m8b-evidence/jobs/fixture/config.json').read_bytes(),config.read_bytes())
                with patch.object(m,'owned',return_value=True),self.assertRaisesRegex(RuntimeError,'Job name exists'):m.main()
                self.assertEqual(popen.call_count,1)


if __name__=='__main__':unittest.main()
