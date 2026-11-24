"""Verify and unpack a pinned model bundle; never loads model code or device APIs."""
from pathlib import Path, PurePosixPath
import hashlib,json,tarfile,time
root=Path('/home/researcher/mea-r2-22554/import_20261116_01')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
summary=json.loads((root/'bundle_summary.json').read_text())
manifest=json.loads((root/'bundle_manifest.json').read_text())
assert summary['archive_sha256']=='46a2fa5f32407141cd9fb403b552b3c141f8ffa5f44f9d66a3567a61251cca20'
archive=root/'model_bundle.tar'
assert archive.stat().st_size==summary['archive_size']
assert sha(archive)==summary['archive_sha256']
assert len(manifest['files'])==summary['file_count']==384
assert sum(x['size'] for x in manifest['files'].values())==summary['total_bytes']
dest=root/'assets'
with tarfile.open(archive,'r:') as tar:
 members=tar.getmembers()
 expected=set(manifest['files'])|{'bundle_manifest.json'}
 assert len(members)==len(expected) and {m.name for m in members}==expected
 for m in members:
  p=PurePosixPath(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  assert (dest/m.name).resolve().is_relative_to(dest.resolve())
 dest.mkdir(exist_ok=False)
 tar.extractall(dest,filter='data')
assert sha(dest/'bundle_manifest.json')==sha(root/'bundle_manifest.json')
verified={}
for name,identity in manifest['files'].items():
 p=dest/name
 assert p.is_file() and not p.is_symlink()
 assert p.stat().st_size==identity['size'] and sha(p)==identity['sha256'],name
 verified[name]=identity['sha256']
base=dest/summary['base_model']
checkpoint=dest/summary['target_checkpoint']
assert sha(base/'config.json')=='b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204'
result=dict(schema='mea.model.transfer.verified.v1',verified_unix=time.time(),source_host='192.0.2.15',target_host='192.0.2.16',
 archive_sha256=summary['archive_sha256'],manifest_sha256=sha(root/'bundle_manifest.json'),
 verified_files=len(verified),verified_bytes=summary['total_bytes'],all_file_hashes_match=True,
 checkpoint_path=str(checkpoint),base_model_path=str(base),source_project_path=str(dest/'m9-research-v5-project'),
 recorded_actor_sha256=summary['actor_sha256'],actor_semantic_hash_recomputed_on_destination=False,
 adapter_file_sha256=sha(checkpoint/'adapter/adapter_model.safetensors'),
 runtime_tested_on_destination=False,training_started=False,source_files_retained=True)
with (root/'verification.json').open('x') as f: json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))

