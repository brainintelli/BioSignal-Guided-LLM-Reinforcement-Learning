import hashlib,json,sys,unittest
from pathlib import Path
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r/'project'))
names=['tests.test_mea_shuffle','tests.test_mea_stim_offline','tests.test_mea_ablation','tests.test_mea_long_ablation','tests.test_mea_replay']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
with (r/'gpu_tests.log').open('x') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),failures=len(result.failures),errors=len(result.errors),
    source_manifest_sha256=hashlib.sha256((r/'project_source_manifest.json').read_bytes()).hexdigest())
(r/'gpu_tests_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if not receipt['passed']:raise SystemExit(1)
