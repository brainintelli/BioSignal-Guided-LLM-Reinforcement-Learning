import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117');dest=r/'project'
if dest.exists():raise FileExistsError('No project overwrite')
with zipfile.ZipFile(r/'project_frozen.zip') as z:
    for i in z.infolist():
        if not (dest/i.filename).resolve().is_relative_to(dest.resolve()):raise ValueError('Zip path')
    z.extractall(dest)
manifest=json.loads((r/'project_source_manifest.json').read_text())
for name,expected in manifest.items():
    if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source mismatch '+name)
for p in json.loads((r/'model_config.json').read_text()).values():
    if not Path(p).is_dir():raise ValueError('Missing original model path')
cmd=[sys.executable,'-m','unittest','tests.test_mea_stim_offline','tests.test_mea_ablation','tests.test_mea_long_ablation','tests.test_mea_replay','-v']
with (r/'gpu_preflight_tests.log').open('x') as log:
    result=subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError('Remote preflight failed; inspect log')
sys.path.insert(0,str(dest))
from agent.mea_stim_offline import StimRecording
pin='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'
sources={c:StimRecording(r/'data',pin=pin,phase=c).summary() for c in ('before','after')}
receipt=dict(passed=True,files=len(manifest),data=sources,source_manifest_sha256=hashlib.sha256((r/'project_source_manifest.json').read_bytes()).hexdigest())
(r/'preflight_passed.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
