"""Register the six-round 22021 physical split, common training budget and held-out games."""
from itertools import groupby
from pathlib import Path
import hashlib
import json
import os
import random
import sys
H = Path(__file__).resolve().parent
W = H.parents[2]
R = H.parent / 'cpp-crossover-v1'
sys.path[:0] = [str(W/'.tools/mea_baseline_deps'), str(W/'standalone/mea_state2_r2')]
os.environ['HDF5_PLUGIN_PATH'] = str(W/'.tools/mea_maxwell_hdf5_plugin')
import h5py
import numpy as np
from mea_state2_r2.features import digest, extract, fit, transform
FS=20000
RECORD=R
def read(p): return json.loads(p.read_bytes())
def save(p,x):
    with p.open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
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

def permute(windows, rng):
    result = []
    for trial, group in groupby(range(len(windows)),key=lambda i:windows[i]['trial']):
        ids = list(group)
        assert len(ids)%2==0 and all(windows[i]['frame_end']==windows[j]['frame_start'] for i,j in zip(ids,ids[1:]))
        blocks = [ids[i:i+2] for i in range(0,len(ids),2)]
        keys = list(range(len(blocks)))
        if len(keys)>1:
            while True:
                rng.shuffle(keys)
                if all(i!=v for i,v in enumerate(keys)):break
        result.extend(i for k in keys for i in blocks[k])
    assert sorted(result)==list(range(len(windows)))
    return result


def main():
    hashes=read(R/'raw-hashes.json'); audited=read(R/'raw-readback-check.json')
    spec=read(R/'design.json'); bind=read(R/'binding.json'); roi=spec['roi']
    assert bind['chip_id']=='22021' and bind['session_id']=='predict22021-20261119T042416Z'
    assert len(roi)==len(set(roi))==324 and len(hashes)==19 and len(audited)==18
    assert read(H.parent/'workspace-readback.json')['raw_readback_matches_device']
    a,b,sp,m=read_record('warmup',hashes,roi)
    assert not m
    norm=fit([window(sp,a,b,a+(12+i)*FS,-i-1,0,12+i,roi)['features'] for i in range(24)])
    train=[];boot=[];test=[]
    for i,trial in enumerate(spec['trials'],1):
        n=f'trial_{i:02d}';ev=read(R/(n+'-evidence.json'));ar=audited[i-1]
        a,b,sp,m=read_record(n,hashes,roi)
        assert m==ev['frames'] and trial==ev['plan'] and ar['chip']==bind['chip_id'] and ar['condition']==trial['condition']
        assert ev['pattern_verified'] and not ev['late_site_rail'] and ev['metadata']['frame_gaps']==0 and ar['index']==i
        assert trial['block']==1+(i-1)//3
        def take(off):
            w=window(sp,a,b,m[0]+off*FS,100*i+off,i,off,roi)
            w['normalized']=list(transform(w['features'],norm));return w
        if i<=12:
            train.extend(take(k) for k in range(8,12))
            if i%3==0:boot.append(take(12))
        else:test.extend(take(k) for k in range(2,12))
    assert list(map(len,[train,boot,test]))==[48,4,60]
    windows=sorted(train+boot+test,key=lambda w:w['frame_start'])
    assert all(u['frame_end']<=v['frame_start'] for u,v in zip(windows,windows[1:]))
    assert max(w['frame_end'] for w in train+boot)<=min(w['frame_start'] for w in test)
    used=set()
    def gather(x):
        if type(x)==int:used.add(x)
        elif isinstance(x,(list,dict)):
            for y in (x.values() if isinstance(x,dict) else x):gather(y)
    for file in (W/'output').rglob('*.json'):
        if file.name not in ('design.json','protocol.json','plan.json','scene-check.json','paired-protocol.json','clean-protocol.json') or H in file.parents:continue
        x=read(file)
        if isinstance(x,dict):
            for key in ('worlds','holdout_seeds','training_seeds','seeds','episodes','restart_worlds','training_worlds','map_seeds','eval_seeds','test_seeds'):
                gather(x.get(key))
    rng=random.Random(220211119042416)
    seeds=rng.sample(range(351000000,352000000),8)
    maps=rng.sample(range(352000000,353000000),64)
    trainmaps=rng.sample(range(353000000,354000000),32)
    assert not set(seeds+maps+trainmaps)&used
    c=dict(chip='22021',sample=bind['organoid_id'],profile='six_block_h5_replay',state2_source='held_separate_device_frame_windows',
           source_sha256=digest(hashes),normalization=norm,roi_sha=digest(roi),series=dict(train=train,bootstrap=boot,holdout=[]))
    tr=dict(study='six_block_stress_22021',session_id=bind['session_id'],chip='22021',arms=['masked','ordered','shuffled'],
        sample=c['sample'],alpha=1,beta=0,feedback_rule='diagnostic only: successful net water gain on drink when starting water <=20',
        starting_scene='supplied_95',starting_reserves={'health':100,'thirst':95,'hunger':95},batch=12,updates=4,eval_decisions=0,
        training_seeds=seeds,episodes={str(seed):trainmaps[j*4:j*4+4] for j,seed in enumerate(seeds)},worlds=maps,horizon=60,
        context_sha=digest(c),normalization=norm,roi_sha=c['roi_sha'],source_sha=c['source_sha256'],physical_training_blocks=[1,2,3,4],
        physical_holdout_blocks=[5,6],calibration='warmup 24 seconds, device frames',windows=test,
        training_order={str(seed):permute(train,random.Random(seed+835)) for seed in seeds},
        shuffle={str(world):permute(test,random.Random(world+839)) for world in maps},
        shuffle_unit='joint two-second blocks deranged inside each individual trial',
        game_scenes=read(H/'stress/scenes.json'),bootstrap_seed=2202111194402,draws=40000,
        primary='tick 60 survival, real versus both controls in each scene, 6 contrasts',
        targeted_secondary='thirst drink and hunger eat restricted waits, 4 contrasts',
        exploration='new cells across three scenes, 6 contrasts',
        stable='Both paired crossed-bootstrap Bonferroni intervals favorable and >=6/8 favorable training seeds separately against each control. Other endpoints descriptive.',
        prior_ids_checked=len(used),training_code_sha=sha(H/'train.py'),eval_code_sha=sha(H/'play60.py'),
        stress_eval_code_sha=sha(H/'stress/evaluate.py'),analysis_code_sha=sha(H/'stress/analyze.py'))
    assert all(len(set(tr['training_order'][str(v)]))==48 for v in seeds)
    save(H/'context.json',c);save(H/'design.json',tr)
    print('Registered',bind['organoid_id'],len(train),len(boot),len(test),len(seeds),len(maps),len(used),digest(tr))
if __name__=='__main__':main()
