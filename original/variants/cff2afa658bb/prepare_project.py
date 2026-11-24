import hashlib,json,zipfile,shutil
from pathlib import Path
r=Path(__file__).resolve().parent;old=r.parent/'mea_stim_offline_20261117';dest=r/'project'
dest.mkdir(exist_ok=False)
manifest=json.loads((old/'project_source_manifest_v2.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(old/'project_frozen_v2.zip') as z:
    if set(z.namelist())!=set(manifest):raise ValueError('Parent inventory')
    for name in manifest:
        p=(dest/name).resolve()
        if not p.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive')
    z.extractall(dest)
for name,h in manifest.items():
    if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=h:raise ValueError('Parent hash: '+name)
shutil.copytree(old/'data',r/'data')
print(json.dumps({'verified_parent_files':len(manifest),'project':str(dest)}))
