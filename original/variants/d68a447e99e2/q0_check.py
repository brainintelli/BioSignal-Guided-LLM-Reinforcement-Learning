import hashlib
import json
from pathlib import Path
import sys

rec = Path('/home/researcher/mea-oddball/oddball22021-20261123/records/oddball-v1')
out = Path('/home/researcher/mea-oddball/oddball22021-20261123/exploratory-v1')
plan = json.loads((rec / 'plan.json').read_text(encoding='utf-8'))
manifest = json.loads((rec / 'manifest.json').read_text(encoding='utf-8'))

main_blocks = [b['name'] for b in plan['blocks'] if b['stage'] == 'main']
assert len(main_blocks) == 12, f'Expected 12 main blocks, got {len(main_blocks)}'

results = []
mismatches = []
for name in main_blocks:
    rel = f'{name}/{name}.raw.h5'
    expected = manifest[rel]
    fpath = rec / rel
    h = hashlib.sha256()
    size = 0
    with fpath.open('rb') as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    digest = h.hexdigest()
    ok = (size == expected['bytes'] and digest == expected['sha256'])
    row = {
        'block': name,
        'rel_path': rel,
        'bytes': size,
        'expected_bytes': expected['bytes'],
        'sha256': digest,
        'expected_sha256': expected['sha256'],
        'match': ok,
    }
    results.append(row)
    print(f"{name:18s}  bytes={size}  match={ok}  sha256={digest[:16]}...")
    if not ok:
        mismatches.append(row)

summary = {'checked_blocks': len(results), 'all_match': len(mismatches) == 0, 'blocks': results}
(out / 'q0_integrity.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if mismatches:
    print('SHA256 MISMATCH DETECTED!', file=sys.stderr)
    sys.exit(2)
print('ALL 12 MAIN BLOCKS MATCH MANIFEST SHA-256.')
