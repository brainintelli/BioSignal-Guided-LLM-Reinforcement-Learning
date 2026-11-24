"""Authenticate archive against SSH-observed hash, then extract exclusively."""
import argparse,hashlib,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('expected_sha256');args=p.parse_args()
assert len(args.expected_sha256)==64 and all(c in '0123456789abcdef' for c in args.expected_sha256)
archive=R/'transfer/evidence.tar';h=hashlib.sha256()
with archive.open('rb') as f:
    for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
assert h.hexdigest()==args.expected_sha256,'Transport checksum mismatch'
destination=R/'evidence';assert not destination.exists()
with tarfile.open(archive,'r') as tar:
    members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk()
        assert (destination/m.name).resolve().is_relative_to(destination.resolve()),m.name
    destination.mkdir();tar.extractall(destination,filter='data')
receipt=dict(archive_sha256=h.hexdigest(),archive_bytes=archive.stat().st_size,extracted_files=len(names),source='SSH/scp 192.0.2.16 dedicated task directory')
with (R/'transport_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
