"""Audit a finalized dual-stream capture without promoting raw progress to event coverage."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.mea_recording import load_h5,sha256
from agent.mea_timing import audit_progress,audit_filtered_progress
from agent.mea_protocol import strict_json,canonical


def rows(path):
    result=[]
    with path.open('rb') as f:
        for line in f:
            if not line.endswith(b'\n') or len(line)>2_000_000: raise ValueError('truncated/oversized log')
            result.append(strict_json(line))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    meta,frames,_=load_h5(root/'baseline.raw.h5')
    event_rows=rows(root/'events.jsonl')
    report=audit_progress(rows(root/'events.jsonl.progress.jsonl'),frames,meta['sample_rate_hz'],event_rows)
    acquisition=json.loads((root/'acquisition.json').read_text())
    event_audit=json.loads((root/'audit.json').read_text())
    report['acquisition_passed']=acquisition['success']
    report['recorded_event_agreement_passed']=event_audit['offline_integrity_passed']
    if event_rows[0]['protocol']=='mea.filtered.transport.v1':
        filtered=audit_filtered_progress(event_rows,frames,meta['sample_rate_hz'],event_audit['offline_integrity_passed'])
        report['filtered_stream']=filtered
        report['live_spike_window_complete']=filtered['live_spike_window_complete']
    report['diagnostic_passed']=all(report[k] for k in ('raw_progress_passed','raw_freshness_passed',
        'receive_clock_fit_passed','wall_clock_stable','acquisition_passed','recorded_event_agreement_passed'))
    if 'filtered_stream' in report:report['diagnostic_passed'] &= report['filtered_stream']['observed_transport_gate_passed']
    report['inputs']={n:sha256(root/n) for n in ('events.jsonl','events.jsonl.progress.jsonl','baseline.raw.h5','acquisition.json','audit.json')}
    with (root/'timing_audit.json').open('x') as f:f.write(canonical(report)+'\n')
    print(canonical({k:v for k,v in report.items() if k!='inputs'}))
    return 0 if report['diagnostic_passed'] else 2


if __name__=='__main__':sys.exit(main())
