"""Run on the GPU host after all registered cells and analysis complete."""
import hashlib
import json
from pathlib import Path
import tarfile


root=Path('/home/researcher/mea-ablation-20261116')
experiment=root/'experiment01'
status=json.loads((experiment/'batch_status.json').read_text())
assert status['complete'] is True and status['passed'] is True and len(status['cells'])==24
assert json.loads((experiment/'analysis.json').read_text())['status']=='complete'
files=[];remote={}
for f in sorted(experiment.rglob('*')):
    if not f.is_file():continue
    if f.suffix in {'.pt','.safetensors'}:
        with f.open('rb') as handle:h=hashlib.file_digest(handle,'sha256').hexdigest()
        remote[f.relative_to(root).as_posix()]={'bytes':f.stat().st_size,'sha256':h}
    else:files.append(f)
(root/'remote_only_files.json').write_text(json.dumps(remote,indent=2))
files += [f for f in root.iterdir() if f.is_file() and
          (f.suffix in {'.json','.log'} or f.name=='project_frozen.zip')]
archive=root/'evidence_experiment01.tar.gz'
with tarfile.open(archive,'x:gz') as handle:
    for f in sorted(set(files)):handle.add(f,arcname=f.relative_to(root).as_posix())
with archive.open('rb') as handle:h=hashlib.file_digest(handle,'sha256').hexdigest()
print('ARCHIVE_BYTES',archive.stat().st_size)
print('ARCHIVE_HASH_A',h[:32]);print('ARCHIVE_HASH_B',h[32:])
print('REMOTE_ONLY_CHECKPOINT_FILES',len(remote))
