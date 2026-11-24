"""Verify the immutable P1 package; --deep also recomputes the H5 audit (h5py)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPECTED_ARCHIVE_SHA256 = '61513afcbc50185a543257563fba1bce3bf9ff3d7a13169dde767c6c97b16907'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
    p.add_argument('--archive', type=Path); p.add_argument('--deep', action='store_true')
    args = p.parse_args(); root = args.root.resolve()
    if args.archive and sha(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError('archive hash differs from source')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != expected['size'] or sha(path) != expected['sha256']:
            raise ValueError('file verification failed: '+name)
    sys.path.insert(0, str(root/'p1_software_20261116_v7'))
    from agent.mea_protocol import ResearchStream
    from agent.mea_recording import load_h5, load_live, audit_recording
    run = root/manifest['accepted_run']; report = json.loads((run/'audit.json').read_text())
    acquire = json.loads((run/'acquisition.json').read_text())
    if not acquire['success'] or not report['offline_integrity_passed']: raise ValueError('run not accepted')
    if acquire['stimulation'] or acquire['alpha'] != 1 or acquire['r2_reward_enabled']:
        raise ValueError('experiment boundary changed')
    stream = ResearchStream(max_spikes=1000000)
    with (run/'recording.jsonl').open('rb') as f:
        for b in iter(lambda:f.read(65536), b''): stream.feed(b)
    stream.finish()
    if stream.session.source != 'recording' or stream.watermark != report['audit_interval'][1] or len(stream.spikes) != report['matched_events']:
        raise ValueError('recording replay differs')
    if args.deep:
        meta, frames, spikes = load_h5(run/'baseline.raw.h5')
        meta.update(current_mapping_before=acquire['mapping_before'], current_mapping_after=acquire['mapping_after'],
                    requested_seconds=acquire['requested_seconds'])
        events, status = load_live(run/'events.jsonl'); status['coordinator_ok'] = acquire['success']
        computed = audit_recording(meta, frames, spikes, events, status)
        if any(value != report[key] for key, value in computed.items()): raise ValueError('recomputed audit differs')
    print(json.dumps(dict(passed=True, files_verified=len(manifest['files']), accepted_run=manifest['accepted_run'],
                          replay_events=len(stream.spikes), deep_audit=args.deep,
                          archive_verified=args.archive is not None)))


if __name__ == '__main__': main()
