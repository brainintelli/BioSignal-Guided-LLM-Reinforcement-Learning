"""Pin the four-arm study to device frames in the audited November 18 chip 22021 crossover."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import h5py
import numpy as np
from mea_state2_r2.features import digest, extract, fit, transform

ROOT = Path(__file__).resolve().parent
RECORD = ROOT.parent / 'cpp-crossover-v2'
OLD = ROOT.parent / 'offline-ppo'
ARMS = ('masked', 'sham', 'regular', 'irregular')
FS = 20000

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def write(p, obj):
    b = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode()
    if p.exists():
        if p.read_bytes() != b:
            raise ValueError(f'Existing study differs at {p}')
    else:
        p.write_bytes(b)

def read_record(name, hashes, roi):
    rel = name + '/' + name + '.raw.h5'
    path = RECORD / rel
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != hashes[rel]['sha256']:
            raise ValueError(f'H5 digest changed at {name}')
    with h5py.File(path, 'r') as f:
        rec = f['/data_store/data0000']
        frames = rec['groups/routed/frame_nos']
        first, last = int(frames[0]), int(frames[-1])
        if len(frames) != last - first + 1:
            raise ValueError('Noncontiguous frames')
        sp = rec['spikes'][:]
        sp = sp[np.isin(sp['channel'], roi) & (sp['frameno'] >= first) & (sp['frameno'] <= last)]
        return first, last, sp, [int(v) for v in rec['events']['frameno']]

def window(sp, first, last, start, idx, trial, offset, roi):
    end = start + FS
    if not first <= start < end <= last + 1:
        raise ValueError('Observation extends beyond raw frames')
    box = sp[(sp['frameno'] >= start) & (sp['frameno'] < end)]
    rows = sorted((int(v['frameno']), int(v['channel']), float(v['amplitude'])) for v in box)
    return dict(index=idx, trial=trial, offset=offset, frame_start=start, frame_end=end,
                count=len(rows), events_sha=digest(rows), features=list(extract(rows, start, end, roi, FS)))

def main():
    prev = read(OLD/'design.json')
    binding = read(RECORD/'binding.json')
    hashes = read(RECORD/'raw-hashes.json')
    verified = read(RECORD/'raw-readback-check.json')
    plan = read(RECORD/'design.json')
    roi = plan['roi']
    assert len(roi)==len(set(roi))==322 and len(hashes)==19
    assert binding['chip_id']==prev['chip']=='22021' and binding['session_id']==prev['session_id']
    assert len(verified)==18 and all(v['chip']=='22021' for v in verified)
    a,b,sp,markers = read_record('warmup',hashes,roi)
    assert not markers
    warm = [window(sp,a,b,a+(12+i)*FS,-i-1,0,12+i,roi) for i in range(24)]
    norm = fit([w['features'] for w in warm])
    series = {arm:dict(train=[], bootstrap=[], holdout=[]) for arm in ARMS[1:]}
    for i,trial in enumerate(plan['trials'],1):
        condition = trial['condition']; name = 'trial_%02d'%i; ev=read(RECORD/(name+'-evidence.json'))
        a,b,sp,markers = read_record(name,hashes,roi)
        assert trial==ev['plan'] and markers==ev['frames'] and verified[i-1]['index']==i
        assert ev['pattern_verified'] and ev['intended_edges_present'] and not ev['late_site_rail']
        assert verified[i-1]['condition']==condition and verified[i-1]['pulses']==len(trial['events'])
        assert ev['metadata']['fs']==FS and ev['metadata']['chip_id']=='22021'
        offsets = list(range(2,15)) if trial['block']<=4 else [2,7,12]
        for offset in offsets:
            key = 'holdout' if trial['block']>4 else ('bootstrap' if offset==14 else 'train')
            w = window(sp,a,b,markers[0]+offset*FS,i*100+offset,i,offset,roi)
            w['normalized']=list(transform(w['features'],norm))
            series[condition][key].append(w)
    for arm,s in series.items():
        assert {k:len(v) for k,v in s.items()}==dict(train=48,bootstrap=4,holdout=6)
        ordered = sorted(s['train']+s['bootstrap']+s['holdout'],key=lambda w:w['frame_start'])
        assert all(x['frame_end']<=y['frame_start'] for x,y in zip(ordered,ordered[1:]))
        assert max(w['trial'] for w in s['bootstrap']+s['train'])<min(w['trial'] for w in s['holdout'])
    assert all(Counter(t['condition'] for t in plan['trials'] if t['block']==b)==Counter(ARMS[1:]) for b in range(1,7))
    source = digest(hashes)
    contexts={}
    for arm in ARMS:
        contexts[arm]=dict(chip='22021',sample=binding['organoid_id'],clock='MaxOne 20 kHz frames',
            normalization=norm,source_sha256=source,roi_sha=digest(roi),profile='22021_input_condition_ablation',
            state2_source='masked' if arm=='masked' else 'device_spikes_'+arm,
            series=series['sham' if arm=='masked' else arm])
    # Distinct seed namespace, fixed before model fitting. Check every available prior design.
    worlds=random.Random(2026111822021).sample(range(172000000,173000000),48)
    historical=set()
    for f in ROOT.parents[1].rglob('design.json'):
        if ROOT in f.parents: continue
        old=read(f)
        if isinstance(old,dict):
            historical.update(v for v in old.get('holdout_seeds',[]) if isinstance(v,int))
    train_worlds={v for seq in prev['episodes'].values() for v in seq}
    assert not set(worlds)&(historical|train_worlds|set(prev['training_seeds']))
    spec={k:prev[k] for k in ('chip','session_id','starting_scene','starting_reserves','training_seeds','episodes','batch','updates','eval_decisions','feedback_stride','minimum_reserve')}
    spec.update(arms=list(ARMS),holdout_seeds=worlds,alpha=1,beta=0,contexts={k:digest(v) for k,v in contexts.items()},
        physical_training_blocks=[1,2,3,4], physical_holdout_blocks=[5,6],source_sha=source,baseline_windows=warm,
        endpoints=['new_cells','r1_sum'],contrasts=[['sham','masked'],['regular','sham'],['irregular','sham']],
        interval=dict(draws=30000,seed=632201,quantiles=[.05/(2*6),1-.05/(2*6)]),
        stable='Both adjusted lower bounds positive and at least 6 of 8 training seeds jointly positive per contrast',
        selection='All arms use the same architecture, initial weights, R1, train seeds, worlds, ticks, horizon and budget. Mask forces all four policy and critic inputs to 0.')
    for arm,obj in contexts.items():write(ROOT/(arm+'.json'),obj)
    write(ROOT/'design.json',spec)
    print('verified source',source,'model arms',len(contexts),'training worlds',len(train_worlds),'prior holdout worlds checked',len(historical))
if __name__=='__main__':main()
