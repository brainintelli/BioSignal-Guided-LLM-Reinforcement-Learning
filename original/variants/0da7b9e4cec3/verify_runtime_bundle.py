from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,shutil,time
root=Path('/home/researcher/mea-r2-22554/runtime_20261116_01')
m=json.loads((root/'runtime_manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'runtime.tar')==m['archive_sha256']
assert (root/'runtime.tar').stat().st_size==m['archive_size']
dest=root/'assets'
with tarfile.open(root/'runtime.tar') as tar:
 members=tar.getmembers()
 assert len(members)==len(m['files']) and {x.name for x in members}==set(m['files'])
 for x in members:
  p=PurePosixPath(x.name)
  assert x.isfile() and not p.is_absolute() and '..' not in p.parts
  assert (dest/x.name).resolve().is_relative_to(dest.resolve())
 dest.mkdir(exist_ok=False)
 tar.extractall(dest,filter='data')
for name,item in m['files'].items():
 p=dest/name
 assert p.stat().st_size==item['size'] and sha(p)==item['sha256'],name
result=dict(verified_files=len(m['files']),verified_bytes=m['total_bytes'],archive_sha256=m['archive_sha256'],
 manifest_sha256=sha(root/'runtime_manifest.json'),all_files_match=True,time_unix=time.time())
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
work=Path('/home/researcher/mea-r2-22554/smoke_20261116_01')
work.mkdir(exist_ok=False)
shutil.copytree('/home/researcher/mea-r2-22554/import_20261116_01/assets/m9-research-v5-project',work/'project')
print(json.dumps(result,indent=2))

