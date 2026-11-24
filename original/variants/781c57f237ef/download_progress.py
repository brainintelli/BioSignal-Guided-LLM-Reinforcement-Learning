"""Read open-handle size; Windows directory metadata can lag while writing."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
p=r/'campaign_delivery.zip.part'
if not p.exists():p=r/'campaign_delivery.zip'
with p.open('rb') as f:f.seek(0,2);n=f.tell()
total=json.loads((r/'archive_receipt.json').read_text())['bytes']
print(json.dumps(dict(file=p.name,bytes=n,total=total,percent=round(100*n/total,1))))
