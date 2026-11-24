import hashlib,json,shutil,zipfile
from pathlib import Path
r=Path(__file__).resolve().parent;project=r/'project';old=r.parent/'mea_stim_offline_20261117'
parent=json.loads((old/'project_source_manifest_v2.json').read_text())
for name,h in parent.items():
    if hashlib.sha256((project/name).read_bytes()).hexdigest()!=h:raise ValueError('Accepted parent mutated: '+name)
names=set(parent)|{'agent/mea_shuffle.py','agent/mea_memory.py','tools/mea_shuffle_train.py','tools/mea_shuffle_campaign.py','tools/mea_shuffle_parallel.py','tests/test_mea_shuffle.py','tests/test_mea_memory.py'}
source={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in sorted(names)}
with (r/'project_source_manifest.json').open('x') as f:json.dump(source,f,indent=2)
with zipfile.ZipFile(r/'project_frozen.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in source:z.write(project/name,name)
config=json.loads((r.parent/'mea_online_20261117/delivery/configs/gpu.unbound.json').read_text())['model_paths']
(r/'model_config.json').write_text(json.dumps(config,indent=2))
from datetime import datetime,timezone
workspace=r.parents[1]
pins={}
for name in ['docs/superpowers/specs/2026-11-17-mea-28080-shuffle-design.md','docs/superpowers/plans/2026-11-17-mea-28080-shuffle.md']:
    pins[name]=hashlib.sha256((workspace/name).read_bytes()).hexdigest()
receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'source_files':len(source),'parent_files_unchanged':len(parent),
    'source_manifest_sha256':hashlib.sha256((r/'project_source_manifest.json').read_bytes()).hexdigest(),
    'project_archive_sha256':hashlib.sha256((r/'project_frozen.zip').read_bytes()).hexdigest(),'documents':pins,'new_model_scores_viewed':False}
receipt['technical_amendment_sha256']=hashlib.sha256((r/'technical_amendment.json').read_bytes()).hexdigest()
(r/'design_pin.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
