"""Final read-only source/runtime audit and explicit evidence archive."""
import hashlib,json,os,tarfile,time
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);RUNTIME=Path(os.environ['TASK_RUNTIME'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
for name in ['terminal01','stability01']:
    for suffix in ['.exit','_verify.exit']:assert (ROOT/(name+suffix)).read_text().strip()=='0'
    assert json.loads((ROOT/name/'independent_verification.json').read_text())['passed']
mp=RUNTIME.parent/'runtime_manifest.json'
assert sha(mp)=='3f71b79f474aa61d4fa7ebd35d0b07f90b2988ddc01e110aeddb8ca7fed82737'
manifest=json.loads(mp.read_text());size=0
for name,entry in manifest['files'].items():
    p=RUNTIME/name;assert p.stat().st_size==entry['size'] and sha(p)==entry['sha256'],name;size+=entry['size']
write(ROOT/'runtime_audit.json',dict(passed=True,files=len(manifest['files']),bytes=size,manifest_sha256=sha(mp),time=time.time()))
names=['base_smoke.py','layout_offload.py','stability_run.py','protocol.py','test_protocol.py','probe_terminal_fixture.py',
    'terminal_fixture.json','package_evidence.py','runtime_audit.json','godot_import.log','execution_environment.json']
for prefix in ['terminal01','stability01']:
    names.extend([prefix+'.log',prefix+'.exit',prefix+'_verify.log',prefix+'_verify.exit'])
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/prefix).rglob('*') if p.is_file())
write(ROOT/'evidence_manifest.json',{name:dict(sha256=sha(ROOT/name),size=(ROOT/name).stat().st_size) for name in sorted(names)})
with tarfile.open(ROOT/'evidence.tar','x') as archive:
    for name in sorted(names+['evidence_manifest.json']):archive.add(ROOT/name,arcname=name,recursive=False)
h=sha(ROOT/'evidence.tar');write(ROOT/'transport.json',dict(sha256=h,size=(ROOT/'evidence.tar').stat().st_size))
print(json.dumps(dict(files=len(names)+1,bytes=(ROOT/'evidence.tar').stat().st_size)))
print(h[:32]);print(h[32:])
