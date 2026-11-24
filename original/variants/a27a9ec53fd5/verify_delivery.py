"""Verify remote-pinned archive bytes, extract safely, and verify all campaign files."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import shutil
import stat
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--size', required=True, type=int)
    args = parser.parse_args()
    for value in (args.archive_sha256, args.manifest_sha256):
        if not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('invalid pinned SHA256')
    root = Path(__file__).resolve().parent
    archive = root / 'campaign_delivery.zip'
    receipt = json.loads((root / 'archive_receipt.json').read_text(encoding='utf-8'))
    assert receipt['passed'] is True and receipt['accepted_cells'] == 24
    assert receipt['archive_size'] == args.size == archive.stat().st_size
    assert receipt['archive_sha256'] == args.archive_sha256 == sha(archive)
    assert receipt['accepted_manifest_sha256'] == args.manifest_sha256
    print('ARCHIVE_SHA256_VERIFIED', args.size, flush=True)
    destination = root / 'delivery'
    if destination.exists():
        raise FileExistsError('preserve the existing delivery; do not overwrite')
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [item.filename for item in members]
        assert len(names) == len(set(name.casefold() for name in names)), 'duplicate/case-colliding paths'
        for item in members:
            name = item.filename
            parts = PurePosixPath(name).parts
            assert name and not name.startswith('/') and '\\' not in name and ':' not in name
            assert all(part not in ('', '.', '..') for part in parts)
            assert not stat.S_ISLNK(item.external_attr >> 16), 'symbolic link'
            target = destination.joinpath(*parts).resolve()
            assert target.is_relative_to(destination.resolve()), 'escaped archive path'
        assert sum(item.file_size for item in members) + 1024**3 < shutil.disk_usage(root).free
        manifest_bytes = bundle.read('accepted_campaign_manifest.json')
        assert hashlib.sha256(manifest_bytes).hexdigest() == args.manifest_sha256
        destination.mkdir()
        bundle.extractall(destination)
    print('EXTRACTED', len(members), flush=True)
    manifest = json.loads((destination / 'accepted_campaign_manifest.json').read_text(encoding='utf-8'))
    assert manifest['accepted_cells'] == 24
    campaign = destination / 'experiment01'
    actual = {p.relative_to(campaign).as_posix() for p in campaign.rglob('*') if p.is_file()}
    assert actual == set(manifest['files']), 'campaign file inventory differs'
    for count, (name, expected) in enumerate(manifest['files'].items(), 1):
        path = (campaign / name).resolve()
        assert path.is_relative_to(campaign.resolve())
        assert path.stat().st_size == expected['size'] and sha(path) == expected['sha256'], name
        if count % 500 == 0:
            print('VERIFIED_CAMPAIGN_FILES', count, flush=True)
    verified = dict(passed=True, verified_utc=datetime.now(timezone.utc).isoformat(),
                    archive_size=args.size, archive_sha256=args.archive_sha256,
                    accepted_manifest_sha256=args.manifest_sha256,
                    verified_campaign_files=len(actual), accepted_cells=24)
    (root / 'transport_verification.json').write_text(json.dumps(verified, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verified), flush=True)


if __name__ == '__main__':
    main()
