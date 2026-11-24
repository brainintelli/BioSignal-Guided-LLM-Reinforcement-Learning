import hashlib,json,shutil,zipfile,sys
from pathlib import Path
r=Path(__file__).resolve().parent;workspace=r.parents[1];old=r.parent/'mea_long_ablation_20261116/readiness'
suffix=sys.argv[1] if len(sys.argv)>1 else ''
if suffix not in ('','_v2'):raise ValueError('Revision')
dest=r/('project'+suffix);dest.mkdir(exist_ok=False)
with zipfile.ZipFile(old/'project_frozen.zip') as z:
    for info in z.infolist():
        p=(dest/info.filename).resolve()
        if not p.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive')
    z.extractall(dest)
manifest=json.loads((old/'project_source_manifest.json').read_text())
for name,h in manifest.items():
    if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=h:raise ValueError('Prior frozen source mismatch')
names=['agent/mea_stim_offline.py','agent/mea_contract.py','tools/mea_stim_export.py','tools/mea_stim_train.py',
       'tools/mea_stim_campaign.py','tests/test_mea_stim_offline.py','tests/test_mea_replay.py']
for name in names:
    p=dest/name;p.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(workspace/name,p)
source={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
(r/('project_source_manifest'+suffix+'.json')).write_text(json.dumps(source,indent=2))
archive=r/('project_frozen'+suffix+'.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for name in source:z.write(dest/name,name)
shutil.copy2(old/'model_config.json',r/'model_config.json')
print(json.dumps({'files':len(source),'archive_bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
