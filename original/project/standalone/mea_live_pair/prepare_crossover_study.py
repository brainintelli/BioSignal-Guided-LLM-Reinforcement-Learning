"""Build disjoint time-varying replay observations from the completed randomized recording."""
import argparse
import hashlib
import json
import math
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

from mea_state2_r2.features import digest, extract, fit, transform

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def prepare(record,protocol):
    session=read(record/'binding.json')
    plan=read(record/'design.json')
    end=read(record/'end.json')
    if session['session_id']!=protocol['session_id'] or session['chip_id']!=protocol['chip']:
        raise ValueError('Wrong sample or chip')
    if end.get('error') or end['cpp_error'] or end['cleanup'] or not end['closed'] or end['completed']!=18:
        raise ValueError('Incomplete acquisition')
    roi=plan['roi']
    allowed=set(roi)
    raw=(record/'cpp-spikes.jsonl').read_bytes()
    spikes=[json.loads(l) for l in raw.splitlines()]
    if len(spikes)!=end['cpp_rows'] or len(allowed)!=len(roi) or len(roi)<50:
        raise ValueError('Return stream or ROI incomplete')
    events=sorted((ms,ch,amp) for ms,ch,amp,_ in spikes if ch in allowed)
    timestamps=[e[0] for e in events]
    ack=[]
    for l in (record/'api_audit.jsonl').read_text(encoding='utf8').splitlines():
        r=json.loads(l)
        if r['kind']=='ack' and r['command'].startswith('sequence_send ') and r['reply']=='OK':
            ack.append(datetime.fromisoformat(r['utc']).timestamp())
    if len(ack)!=18 or len(plan['trials'])!=18:
        raise ValueError('Incomplete trial sequence')
    def window(t,idx,trial,offset):
        start=math.ceil(t*1000)
        if timestamps[0]>=start or timestamps[-1]<start+1000:
            raise ValueError('Spike stream does not cover observation')
        rows=events[bisect_left(timestamps,start):bisect_left(timestamps,start+1000)]
        return dict(index=idx,frame_start=start,frame_end=start+1000,trial=trial,offset=offset,
                    count=len(rows),features=list(extract(rows,start,start+1000,roi,1000)),events_sha=digest(rows))
    baseline=[window(ack[0]-30+i,-i-1,0,i)['features'] for i in range(24)]
    norm=fit(baseline)
    series=dict(train=[],bootstrap=[],holdout=[])
    for i,(t,spec) in enumerate(zip(ack,plan['trials']),1):
        receipt=read(record/('trial_%02d-evidence.json'%i))
        if (receipt['metadata']['chip_id']!=session['chip_id'] or receipt['metadata']['frame_gaps'] or
            not receipt['pattern_verified'] or receipt['late_site_rail'] or receipt['plan']!=spec or
            not 0<=receipt['ack_end']-t<1 or spec['index']!=i):
            raise ValueError('Unverified trial or timestamp')
        if i<=12:
            for offset in range(8,12):
                series['train'].append(window(t+offset,(i-1)*4+offset-7,i,offset))
            if i%3==0:
                series['bootstrap'].append(window(t+12,100+i,i,12))
        else:
            offset=8+(i-13)%4
            series['holdout'].append(window(t+offset,200+i,i,offset))
    all_windows=sorted(sum(series.values(),[]),key=lambda w:w['frame_start'])
    if any(a['frame_end']>b['frame_start'] for a,b in zip(all_windows,all_windows[1:])):
        raise ValueError('Overlapping registered observations')
    for w in all_windows:
        w['normalized']=list(transform(w['features'],norm))
    return dict(chip=session['chip_id'],sample=session['organoid_id'],clock='C++ host milliseconds',
                source_sha256=hashlib.sha256(raw).hexdigest(),profile='randomized_crossover_game_replay',
                state2_source='current_crossover_cpp_spike_stream',roi_sha=digest(roi),
                normalization=norm,series=series,device_ack_count=len(ack))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('record',type=Path)
    parser.add_argument('study',type=Path)
    args=parser.parse_args()
    spec=read(args.study/'protocol.json')
    ctx=prepare(args.record,spec)
    for n,obj in [('context',ctx),('design',spec|{'context_sha':digest(ctx)})]:
        with (args.study/(n+'.json')).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    print('Source windows',{k:len(v) for k,v in ctx['series'].items()})
