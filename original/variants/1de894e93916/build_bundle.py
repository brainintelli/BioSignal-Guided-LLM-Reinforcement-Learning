"""Seal a new source/data bundle without modifying previous experiments."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
name=sys.argv[1]
if not name.isalnum():raise ValueError('Simple bundle name required')
files={}
old=root/'output/mea_28080_shuffle_20261117_v2'
original=json.loads((old/'project_source_manifest.json').read_text())
for relative,expected in original.items():
    payload=(old/'project'/relative).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Frozen project changed: '+relative)
    files['project/'+relative]=payload
for package in ('mea_state2_r2','mea_binary_r2'):
    source=root/'standalone'/package
    for path in source.rglob('*.py'):
        if '__pycache__' not in path.parts:
            files['project/standalone/'+package+'/'+path.relative_to(source).as_posix()]=path.read_bytes()
for path in (old/'data').iterdir():
    if path.is_file():files['data/'+path.name]=path.read_bytes()
files['model_config.json']=(old/'model_config.json').read_bytes()
files['stimulus_profile.json']=(root/'configs/mea_binary_r2_22021_20261118.json').read_bytes()
files['external_assets.json']=(out/'external_assets.json').read_bytes()
manifest={name:hashlib.sha256(value).hexdigest() for name,value in sorted(files.items())}
files['manifest.json']=json.dumps(manifest,indent=2).encode()
path=out/('bundle_'+name+'.zip')
with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for relative,payload in files.items():archive.writestr(relative,payload)
receipt=dict(path=str(path),files=len(manifest),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(files['manifest.json']).hexdigest())
with (out/('bundle_'+name+'.json')).open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
