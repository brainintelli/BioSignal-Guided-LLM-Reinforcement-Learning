import json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
class DeploymentTests(unittest.TestCase):
    def test_extract_into_pythonpath_refreshes_import_cache(self):
        source=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ('remote_prepare.py','project_frozen.zip','project_source_manifest.json'):
                shutil.copy2(source/name,root/name)
            shutil.copytree(source/'data',root/'data')
            (root/'model_config.json').write_text(json.dumps({'model':str(root)}))
            env=dict(os.environ,PYTHONPATH=str(root/'project'),PYTHONDONTWRITEBYTECODE='1')
            result=subprocess.run([sys.executable,str(root/'remote_prepare.py')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([sys.executable,str(root/'remote_prepare.py'),'--verify-existing'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            (root/'project/agent/mea_shuffle.py').write_text('corrupt')
            result=subprocess.run([sys.executable,str(root/'remote_prepare.py'),'--verify-existing'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Source agent/mea_shuffle.py',result.stderr)
if __name__=='__main__':unittest.main()
