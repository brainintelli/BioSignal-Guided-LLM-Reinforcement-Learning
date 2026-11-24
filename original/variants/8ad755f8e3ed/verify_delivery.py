"""Check archive and every sealed file before any report generation."""
import hashlib,json,zipfile
from pathlib import Path
r=Path(__file__).resolve().parent;receipt=json.loads((r/'archive_receipt.json').read_text());archive=r/'campaign_delivery.zip'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if archive.stat().st_size!=receipt['bytes'] or sha(archive)!=receipt['archive_sha256']:raise RuntimeError('Archive pin mismatch')
dest=r/'delivery';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    names=[i.filename for i in z.infolist()]
    if len(set(names))!=len(names):raise RuntimeError('Duplicate zip members')
    for i in z.infolist():
        if not (dest/i.filename).resolve().is_relative_to(dest.resolve()):raise RuntimeError('Escaping zip member')
    z.extractall(dest)
manifest=dest/'accepted_campaign_manifest.json'
if sha(manifest)!=receipt['accepted_manifest_sha256']:raise RuntimeError('Accepted manifest mismatch')
m=json.loads(manifest.read_text())
if m.get('passed') is not True or m.get('cells')!=24 or len(names)!=receipt['files'] or set(names)!=set(m['files'])|{'accepted_campaign_manifest.json'}:raise RuntimeError('Unexpected delivery inventory')
for name,q in m['files'].items():
    p=(dest/name).resolve()
    if not p.is_relative_to(dest.resolve()) or p.stat().st_size!=q['bytes'] or sha(p)!=q['sha256']:raise RuntimeError('Sealed file mismatch: '+name)
for name in ['project_frozen.zip','project_source_manifest.json']:
    own=r/('project_frozen_v2.zip' if name.endswith('.zip') else 'project_source_manifest_v2.json')
    if sha(dest/name)!=sha(own):raise RuntimeError('Final source differs from frozen local original')
result=dict(passed=True,files=len(m['files']),archive_sha256=receipt['archive_sha256'],accepted_manifest_sha256=receipt['accepted_manifest_sha256'],records=m['records'])
(r/'delivery_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
