"""Export completed campaign evidence, retaining binary checkpoints on the GPU host."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


root=Path(sys.argv[1]).resolve()
status=json.loads((root/'campaign_status.json').read_text())
if status.get('complete') is not True or status.get('passed') is not True or status.get('accepted_cells')!=40:
    raise ValueError('Only a complete accepted 40-cell campaign can be exported')
archive=root.parent/(root.name+'_evidence.zip')
manifest={};checkpoints={};paths=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    relative=p.relative_to(root).as_posix()
    if relative.split('/')[0] in ('project','data'):continue
    if p.suffix in ('.pt','.safetensors','.bin'):
        checkpoints[relative]={'size':p.stat().st_size,'sha256':sha(p)}
    else:
        if p.is_symlink():raise ValueError('Unexpected symlink in evidence')
        manifest[relative]={'size':p.stat().st_size,'sha256':sha(p)}
        paths.append((relative,p))
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in paths:z.write(p,name)
    z.writestr('checkpoint_inventory.json',json.dumps(checkpoints,indent=2))
    z.writestr('evidence_manifest.json',json.dumps(manifest,indent=2))
receipt={'archive':str(archive),'size':archive.stat().st_size,'sha256':sha(archive),
         'files':len(manifest),'checkpoints':len(checkpoints),'registration_sha256':status['registration_sha256']}
with (root.parent/(root.name+'_export_receipt.json')).open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
