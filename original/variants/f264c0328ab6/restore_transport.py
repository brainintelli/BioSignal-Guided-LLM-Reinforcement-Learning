"""Verify every transport part and restore the original byte-identical tar."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'transport_deps'))
import zstandard
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    m=json.loads((ROOT/'transport_compressed.json').read_text());original=json.loads((ROOT/'transport.json').read_text())
    assert [p['name'] for p in m['parts']]==[f'evidence.tar.zst.part-{i:02d}' for i in range(8)]
    for part in m['parts']:
        p=ROOT/part['name'];assert p.stat().st_size==part['size'] and sha(p)==part['sha256'],part['name']
    compressed=ROOT/'evidence.tar.zst'
    with compressed.open('xb') as out:
        for part in m['parts']:
            with (ROOT/part['name']).open('rb') as src:
                for chunk in iter(lambda:src.read(8*1024**2),b''):out.write(chunk)
    assert compressed.stat().st_size==m['size'] and sha(compressed)==m['sha256']
    restored=ROOT/'evidence.tar'
    with compressed.open('rb') as src,restored.open('xb') as out:
        with zstandard.ZstdDecompressor(max_window_size=1024**3).stream_reader(src) as reader:
            for chunk in iter(lambda:reader.read(8*1024**2),b''):out.write(chunk)
    assert restored.stat().st_size==original['size'] and sha(restored)==original['sha256']
    result=dict(passed=True,parts=8,zstandard_version=zstandard.__version__,compressed_bytes=m['size'],compressed_sha256=m['sha256'],restored_bytes=original['size'],restored_sha256=original['sha256'],partial_uncompressed_transfer_preserved=True)
    with (ROOT/'compressed_transport_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
