from pathlib import Path
import hashlib,json,tarfile,time
root=Path('/home/researcher/meadowmind-gpu')
out=root/'transfers/mea_22554_runtime_20261116_01'
out.mkdir(exist_ok=False)
sources={'python':root/'python/cpython-3.12.16-linux-x86_64-gnu',
'site':root/'m6-venv/lib/python3.12/site-packages',
'overlay':root/'m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site',
'godot':root/'godot'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={}
with tarfile.open(out/'runtime.tar','w',dereference=True) as tar:
 for prefix,folder in sources.items():
  for p in sorted(folder.rglob('*')):
   if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
   if prefix=='godot' and p.name!='Godot_v4.7.2-stable_linux.x86_64':continue
   assert any(p.resolve().is_relative_to(q.resolve()) for q in sources.values()),str(p)
   name=prefix+'/'+p.relative_to(folder).as_posix()
   h=sha(p);size=p.stat().st_size
   tar.add(p,arcname=name,recursive=False)
   assert sha(p)==h and p.stat().st_size==size
   files[name]=dict(size=size,sha256=h)
manifest=dict(source_host='192.0.2.15',created_unix=time.time(),sources={k:str(v) for k,v in sources.items()},files=files,
 file_count=len(files),total_bytes=sum(f['size'] for f in files.values()),archive_sha256=sha(out/'runtime.tar'),archive_size=(out/'runtime.tar').stat().st_size)
(out/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','sources')},indent=2))

