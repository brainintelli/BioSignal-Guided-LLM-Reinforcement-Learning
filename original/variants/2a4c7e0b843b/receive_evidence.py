"""Verify transport before extracting regular files into a new evidence directory."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    archive=ROOT/'evidence.tar';transport=json.loads((ROOT/'transport.json').read_text());out=ROOT/'evidence'
    assert archive.stat().st_size==transport['size'] and sha(archive)==transport['sha256']
    out.mkdir(exist_ok=False)
    with tarfile.open(archive,'r') as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})
        for m in members:
            p=(out/m.name).resolve();assert m.isfile() and p.is_relative_to(out.resolve())
            p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,p.open('xb') as dst:
                for chunk in iter(lambda:src.read(8*1024**2),b''):dst.write(chunk)
            assert p.stat().st_size==m.size
    manifest=json.loads((out/'evidence_manifest.json').read_text())
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    assert actual==set(manifest)|{'evidence_manifest.json'}
    for n,v in manifest.items():
        p=out/n;assert p.stat().st_size==v['size'] and sha(p)==v['sha256'],n
    result=dict(passed=True,archive_bytes=transport['size'],archive_sha256=transport['sha256'],files=len(actual))
    with (ROOT/'transport_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
