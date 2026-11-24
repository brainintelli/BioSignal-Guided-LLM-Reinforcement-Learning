"""Verify archived bytes only; GPU replay/reload receipts are separate evidence."""
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'delivery_manifest.json').read_text(encoding='utf-8'))
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*')
                    if p.is_file() and p.name != 'delivery_manifest.json'
                    and '__pycache__' not in p.parts}
    assert actual_files == set(manifest['files']), 'delivery inventory changed'
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root), name
        assert path.stat().st_size == expected['bytes'], name
        assert sha(path) == expected['sha256'], name
    sources_verified = {}
    for run_name, archive in [
        ('scripted_run03', 'source_accepted_local_run03.zip'),
        ('gpu/model_run03', 'gpu/project_accepted_run03.zip'),
    ]:
        run = root / run_name
        run_manifest = json.loads((run / 'manifest.json').read_text())
        for name, expected in run_manifest['files'].items():
            path = (run / name).resolve()
            assert path.is_relative_to(run), name
            assert sha(path) == expected, name
        with zipfile.ZipFile(root / archive) as snapshot:
            for name, expected in run_manifest['sources'].items():
                data = snapshot.read(name.split('/', 1)[1])
                assert hashlib.sha256(data).hexdigest() == expected, name
        sources_verified[run_name] = len(run_manifest['sources'])
    for name in ['gpu/status_run03.json', 'scripted_run03_replay.json',
                 'independent_feature_audit.json', 'transfer_verification.json',
                 'gpu/model_run03_independent_reload.json',
                 'gpu/model_run03_native_replay.json']:
        assert json.loads((root / name).read_text())['passed'] is True, name
    print(json.dumps(dict(passed=True, verification='archived_bytes_only',
                         files=len(actual_files), sources=sources_verified)))


if __name__ == '__main__':
    main()
