"""Run with vendor Python -s on a finalized P1 acquisition directory."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.mea_recording import audit_recording, load_h5, load_live, export_recording, sha256
from agent.mea_protocol import canonical


def main():
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path)
    args = p.parse_args(); root = args.directory
    acquire = json.loads((root/'acquisition.json').read_text())
    paths = list(root.glob('*.h5'))
    if len(paths) != 1: raise ValueError('expected one finalized H5')
    meta, frames, spikes = load_h5(paths[0])
    meta.update(current_mapping_before=acquire['mapping_before'],
                current_mapping_after=acquire.get('mapping_after'))
    events, status = load_live(root/'events.jsonl')
    status['coordinator_ok'] = acquire['success']
    report = audit_recording(meta, frames, spikes, events, status)
    if meta['chip_id'] != acquire['expected_chip_id']:
        report['errors'].append('chip_id_mismatch'); report['offline_integrity_passed'] = False
    report['metadata'] = meta
    report['inputs'] = {v.name: sha256(v) for v in (paths[0], root/'events.jsonl', root/'acquisition.json')}
    report['limits'] = ['Offline recorded events only; no live completeness claim.',
                        'Raw frame continuity does not establish spike detection sensitivity.',
                        'Biological sample identity missing; no efficacy or R2 interpretation.',
                        'Signed H5 native amplitude multiplied by recorded lsb_volts for uV.']
    if report['offline_integrity_passed']:
        export_recording(root/'recording.jsonl', meta, spikes, report, root.name)
    with (root/'audit.json').open('x', encoding='utf-8') as f: f.write(canonical(report)+'\n')
    print(canonical({k: v for k, v in report.items() if k not in ('metadata', 'inputs')}))
    return 0 if report['offline_integrity_passed'] else 2


if __name__ == '__main__': sys.exit(main())
