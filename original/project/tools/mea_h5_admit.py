"""Offline re-audit only; original rejected audit and source files stay intact."""
import argparse
from pathlib import Path
from agent.mea_recording import load_h5,load_live,audit_recording,export_recording,sha256
from agent.mea_h5_replay import SCHEMA,validate_confirmation
from agent.mea_protocol import canonical,strict_json


def read_json(path):return strict_json(path.read_text(encoding='utf-8'))


def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n')


def _same(actual,expected,label):
    if canonical(actual)!=canonical(expected):raise ValueError(label+' differs')


def examine(run,chip_id):
    acquisition=read_json(run/'acquisition.json');confirmation=read_json(run/'chip_confirmation.json')
    meta,frames,spikes=load_h5(run/'baseline.raw.h5')
    validate_confirmation(acquisition,meta,confirmation,chip_id)
    meta.update(current_mapping_before=acquisition['mapping_before'],current_mapping_after=acquisition['mapping_after'],
        requested_seconds=acquisition['requested_seconds'])
    events,status=load_live(run/'events.jsonl');status['coordinator_ok']=acquisition['success']
    report=audit_recording(meta,frames,spikes,events,status,mapping_policy='finalized_h5_recorded_subset')
    report.update(schema=SCHEMA,metadata=meta,confirmed_chip_id=chip_id,
        inputs={n:sha256(run/n) for n in ('baseline.raw.h5','events.jsonl','acquisition.json','audit.json','chip_confirmation.json')},
        limits=['Original strict audit retained, not changed to passed.',
                'Mapping authority is finalized H5 metadata; no independent hardware readback.',
                'Only deterministic intersection of mapped and raw-saved channels is used.',
                'No stimulation, R2 telemetry only, alpha1; no live completeness or biological efficacy claim.'])
    return report,spikes


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['admit','verify']);p.add_argument('run',type=Path);p.add_argument('--chip-id',required=True)
    a=p.parse_args();report,spikes=examine(a.run,a.chip_id)
    if a.action=='verify':
        _same(report,read_json(a.run/'admission.json'),'independently recomputed H5 admission')
        # Re-export to an exclusive temporary directory; compare every output byte.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'recording.jsonl';export_recording(out,report['metadata'],spikes,report,a.run.name)
            _same(sha256(out),sha256(a.run/'recording.jsonl'),'independently recomputed recording export')
    else:
        write_json(a.run/'admission.json',report)
        if report['offline_integrity_passed']:export_recording(a.run/'recording.jsonl',report['metadata'],spikes,report,a.run.name)
    print({k:report[k] for k in ('offline_integrity_passed','errors','raw_duration_seconds','raw_discontinuities','mapped_channels','matched_events')},flush=True)
    if not report['offline_integrity_passed']:raise SystemExit(2)


if __name__=='__main__':main()
