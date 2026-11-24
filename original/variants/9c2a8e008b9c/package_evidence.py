from pathlib import Path
import json,subprocess,time,tarfile,hashlib
w=Path('/home/researcher/mea-r2-22554/smoke_20261116_01')
assert json.loads((w/'run05_exit.json').read_text())['exit_code']==0
r=json.loads((w/'run05/independent_reload.json').read_text())
assert r['passed']
q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert 'GPU-e9739564-e2ec-59c3-10f7-096d780abcbc' not in q
(w/'gpu_release.json').write_text(json.dumps(dict(time_unix=time.time(),selected_gpu='GPU-e9739564-e2ec-59c3-10f7-096d780abcbc',selected_gpu_no_compute_process=True,all_compute_apps=q),indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={p.name:p for p in w.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.json','.jsonl','.log'] and p.name not in ['evidence_receipt.json']}
for number in range(1,6):
 for p in (w/('run%02d'%number)).rglob('*'):
  if p.is_file():files[p.relative_to(w).as_posix()]=p
root=Path('/home/researcher/mea-r2-22554/runtime_20261116_01')
for name in ['runtime_manifest.json','verification.json']:files['runtime/'+name]=root/name
m=json.loads(Path('/home/researcher/mea-r2-22554/import_20261116_01/assets/bundle_manifest.json').read_text())
for name,entry in m['files'].items():
 prefix='m9-research-v5-project/'
 if name.startswith(prefix):
  rel=name[len(prefix):];p=w/'project'/rel
  assert sha(p)==entry['sha256']
  files['project/'+rel]=p
identities={name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in files.items()}
with tarfile.open(w/'evidence.tar','x') as tar:
 for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
receipt=dict(file_count=len(files),files=identities,archive_sha256=sha(w/'evidence.tar'),archive_size=(w/'evidence.tar').stat().st_size,created_unix=time.time())
(w/'evidence_receipt.json').write_text(json.dumps(receipt,indent=2))
print('package',len(files),receipt['archive_size'])
print(receipt['archive_sha256'][:32]);print(receipt['archive_sha256'][32:])
print(json.dumps(r,indent=2))

