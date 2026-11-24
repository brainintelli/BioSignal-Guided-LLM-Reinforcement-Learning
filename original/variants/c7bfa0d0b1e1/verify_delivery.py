"""Verify timing package, replay, transport prefix and measured clock evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(line) for line in path.read_bytes().splitlines()]
def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--archive',type=Path);p.add_argument('--archive-sha256')
    p.add_argument('--deep',action='store_true');args=p.parse_args();root=args.root.resolve()
    require(bool(args.archive)==bool(args.archive_sha256),'archive and trusted hash must be paired')
    if args.archive:require(sha(args.archive)==args.archive_sha256,'archive hash mismatch')
    m=read(root/'manifest.json')
    actual={v.relative_to(root).as_posix() for v in root.rglob('*') if v.is_file() and v!=root/'manifest.json'}
    require(actual==set(m['files']),'manifest inventory mismatch')
    for name,item in m['files'].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root) and path.stat().st_size==item['size'] and sha(path)==item['sha256'],'file mismatch: '+name)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root/m['final_software']))
    from agent.mea_protocol import ResearchStream
    from agent.mea_recording import load_h5,load_live,audit_recording
    from agent.mea_timing import FilteredFrameStream,audit_progress,audit_filtered_progress,clock_bounds
    run=root/m['accepted_run'];a=read(run/'acquisition.json');r=read(run/'audit.json');t=read(run/'timing_audit.json')
    require(a['success'] and a['legacy_restored'] and r['offline_integrity_passed'] and t['diagnostic_passed'],'capture/audit failure')
    require(a['stimulation'] is False and a['alpha']==1 and a['r2_reward_enabled'] is False,'experiment boundary changed')
    require(t['live_spike_window_complete'] is False and t['filtered_stream']['spike_peak_window_complete'] is False,'unsupported peak coverage claim')
    replay=ResearchStream(max_spikes=1000000)
    with (run/'recording.jsonl').open('rb') as f:
        for chunk in iter(lambda:f.read(65536),b''):replay.feed(chunk)
    replay.finish()
    require(replay.session.source=='recording' and replay.watermark==r['audit_interval'][1] and len(replay.spikes)==r['matched_events'],'offline replay mismatch')
    gate=FilteredFrameStream()
    with (run/'events.jsonl').open('rb') as f:
        for chunk in iter(lambda:f.read(65536),b''):gate.feed(chunk)
    gate.finish();event_rows=lines(run/'events.jsonl')
    require(gate.watermark==t['filtered_stream']['covered_end_frame'],'transport prefix mismatch')
    watch=lines(run/'live_watch.jsonl');w=watch[-1];updates=watch[:-1]
    require(w['passed'] and w['watermark_timebase']=='transport_frame' and w['spike_peak_window_complete'] is False,'watcher failure/timebase')
    require(w['final_watermark']==gate.watermark and w['progress_updates']==len(updates),'watcher count/frontier mismatch')
    before_end=sum(x['observed_mono_raw_ns']<event_rows[-1]['end_mono_ns'] for x in updates)
    require(before_end>0 and all(x['eligible'] and 0<=x['age_ms']<=250 for x in updates),'no timely live prefix observations')
    clock=root/m['clock_run'];c=read(clock/'clock_audit.json');computed=clock_bounds(lines(clock/'samples.jsonl'))
    require(c['success'] and all(c[k]==v for k,v in computed.items()),'host clock evidence differs')
    if args.deep:
        meta,frames,spikes=load_h5(run/'baseline.raw.h5')
        meta.update(current_mapping_before=a['mapping_before'],current_mapping_after=a['mapping_after'],requested_seconds=a['requested_seconds'])
        events,status=load_live(run/'events.jsonl');status['coordinator_ok']=a['success']
        calculated=audit_recording(meta,frames,spikes,events,status)
        require(all(r[k]==v for k,v in calculated.items()),'recomputed H5 event audit differs')
        raw=audit_progress(lines(run/'events.jsonl.progress.jsonl'),frames,meta['sample_rate_hz'],event_rows)
        require(all(t[k]==v for k,v in raw.items()),'recomputed raw timing differs')
        filtered=audit_filtered_progress(event_rows,frames,meta['sample_rate_hz'],r['offline_integrity_passed'])
        require(filtered==t['filtered_stream'],'recomputed filtered timing differs')
    print(json.dumps(dict(passed=True,files_verified=len(m['files']),accepted_run=m['accepted_run'],
        replay_events=len(replay.spikes),transport_frames=gate.frame_count,live_updates_before_capture_end=before_end,
        spike_peak_window_complete=False,clock_samples=c['samples'],deep_audit=args.deep,archive_verified=bool(args.archive)),indent=2))


if __name__=='__main__':main()
