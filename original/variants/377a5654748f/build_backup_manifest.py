"""Finalize the manifest only after all required raw copies verified."""
import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
required=[('recording_resume1','identity.raw.h5'),('recording_resume2','identity.raw.h5'),('recording_resume2','baseline.raw.h5'),
          ('recording_resume2','stimulus_post.raw.h5'),('multi_bounded_recording','identity.raw.h5'),
          ('multi_bounded_resume1','identity.raw.h5'),('multi_bounded_resume2','identity.raw.h5')]
rows=[]
for folder,name in required:
    f=r/'backup'/folder/name
    q=json.loads((f.parent/(f.name+'.verified.json')).read_text())
    if not q['source_verified'] or f.stat().st_size!=q['bytes']:raise RuntimeError('Backup is not verified: '+str(f))
    rows.append(q)
first=r/'identity_initial.raw.h5';h=hashlib.sha256()
with first.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
expected='9a0f67895e23379d5855bbce03a091e784978b291fed2e3f5ef36914d5a9ab48'
if h.hexdigest()!=expected or first.stat().st_size!=417226908:raise RuntimeError('Initial identity copy changed')
rows.append(dict(source='/home/researcher/mea_full_28080_20261117_1406/recording/identity.raw.h5',local=str(first),
                 bytes=first.stat().st_size,sha256=expected,source_verified=True))
out=dict(status='all_8_raw_files_backed_up_and_source_SHA256_verified',total_bytes=sum(x['bytes'] for x in rows),files=rows,
         true_stimulation_record='recording_resume2/stimulus_post.raw.h5',additional_true_stimulation=False,
         header_only_failed_record='multi_bounded_recording/identity.raw.h5',training_admitted=False)
(r/'backup_manifest.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))
