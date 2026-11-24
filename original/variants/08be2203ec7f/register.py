"""Seal an out-of-record replication on new matched games before any model rollout."""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
SESSION=TRAIN.parent
WORK=SESSION.parents[1]
ARCH=SESSION/'offline-mea-stim-ablation'
sys.path[:0]=[str(WORK/'.tools/mea_baseline_deps'),str(WORK/'standalone/mea_state2_r2'),str(ARCH)]
import h5py
import numpy as np
from build_inputs import window,FS
from mea_state2_r2.features import digest,transform

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    old=read(TRAIN/'design.json');ctx=read(TRAIN/'context.json')
    saved=read(ARCH/'archive-frozen-60tick/design.json');crossover=SESSION/'cpp-crossover-v2'
    quiet=SESSION/'baseline/quiet.raw.h5';warm=crossover/'warmup/warmup.raw.h5'
    binding=read(SESSION/'baseline-binding.json')
    assert old['chip']==saved['chip']==binding['chip_id']=='22021' and binding['organoid_id']==saved['sample']==old['sample']
    assert sha(quiet)==read(SESSION/'baseline/verified.json')['sha256']==saved['source_sha']
    assert sha(warm)==read(crossover/'raw-hashes.json')['warmup/warmup.raw.h5']['sha256']
    roi=read(crossover/'design.json')['roi'];assert len(set(roi))==322 and digest(roi)==ctx['roi_sha']
    with h5py.File(quiet) as f,h5py.File(warm) as w:
        r=f['data_store/data0000'];rt=w['data_store/data0000'];assert len(r['events'])==0
        for k in ('gain','hpf','sampling','lsb','spike_threshold'):
            assert np.array_equal(r['settings/'+k][:],rt['settings/'+k][:])
        assert float(r['settings/sampling'][0])==FS
        frames=r['groups/routed/frame_nos'][:].astype('int64');first,last=int(frames[0]),int(frames[-1])
        assert np.all(np.diff(frames)==1) and last<int(rt['groups/routed/frame_nos'][0])
        oldmap={int(q['electrode']):q for q in r['settings/mapping'][:]}
        newmap={int(q['channel']):q for q in rt['settings/mapping'][:]}
        remap={};electrodes=[]
        for ch in roi:
            e=int(newmap[ch]['electrode']);row=oldmap[e]
            assert all(row[k]==newmap[ch][k] for k in ['x','y'])
            remap[int(row['channel'])]=ch;electrodes.append(e)
        assert len(remap)==len(set(electrodes))==322 and electrodes==saved['aligned_electrodes']
        raw=r['spikes'][:];sp=raw[np.isin(raw['channel'],list(remap))].copy()
        sp['channel']=[remap[int(k)] for k in sp['channel']]
        wins=[]
        for i in range(60):
            t=i+2;q=window(sp,first,last,first+t*FS,20000+i,0,t,roi)
            q['normalized']=list(transform(q['features'],ctx['normalization']));wins.append(q)
    assert wins==saved['windows'] and saved['normalization']==ctx['normalization']
    assert last<min(w['frame_start'] for w in ctx['series']['train']+old['windows'])
    seen=set()
    def integers(x):
        if isinstance(x,int):seen.add(x)
        elif isinstance(x,list):
            for v in x:integers(v)
        elif isinstance(x,dict):
            for v in x.values():integers(v)
    for nm in ('design.json','protocol.json','plan.json'):
        for f in (WORK/'output').rglob(nm):
            if HERE in f.parents:continue
            d=read(f)
            if isinstance(d,dict):
                for k in ('worlds','holdout_seeds','training_seeds','seeds','episodes','restart_worlds','training_worlds'):
                    if k in d:integers(d[k])
    worlds=random.Random(22021111964).sample(range(244000000,245000000),64)
    assert not set(worlds)&seen and len(set(worlds))==64
    perms={}
    for world in worlds:
        rng=random.Random(world+22021111965);perm=[]
        for j in range(0,60,20):
            original=list(range(10))
            while True:
                a=rng.sample(original,10)
                if all(i!=v for i,v in enumerate(a)):break
            perm.extend(j+2*b+k for b in a for k in (0,1))
        assert sorted(perm)==list(range(60)) and all(i//20==k//20 and i!=k for i,k in enumerate(perm))
        perms[str(world)]=perm
    conditions={'ordered':['ordered','ordered'],'masked':['masked','masked'],'shuffled':['shuffled','shuffled'],
                'same_masked':['ordered','masked'],'same_shuffled':['ordered','shuffled']}
    audit=read(TRAIN/'training-audit.json')
    assert len(audit)==24 and all(r['reward_contribution']==0 and r['steps']==48 for r in audit)
    files={f"{a['seed']}/{a['arm']}":a['after'] for a in audit}
    spec=dict(chip=old['chip'],sample=old['sample'],source_sha=sha(quiet),source_file='baseline/quiet.raw.h5',source_frames=[first,last],
        physical_electrodes=electrodes,remap={str(k):v for k,v in remap.items()},roi_sha=ctx['roi_sha'],normalization=ctx['normalization'],
        previous_study_sha=digest(old),calibration_file_sha=sha(warm),training_audit_sha=sha(TRAIN/'training-audit.json'),
        previous_play_sha=sha(TRAIN/'play60.py'),runner_sha=sha(HERE/'evaluate.py'),states=files,conditions=conditions,
        training_seeds=old['training_seeds'],worlds=worlds,prior_map_ids_checked=len(seen),shuffle=perms,windows=wins,horizon=60,
        starting_scene=old['starting_scene'],starting_reserves=old['starting_reserves'],reward='R1 only; alpha=1, R2 log only; frozen eval',
        main=['masked','shuffled'],mechanism=['same_masked','same_shuffled'],primary_endpoint='new_cells',
        secondary_endpoints=['r1','survived'],draws=40000,resampling_seed=22021111966,
        decision='ordered minus control, separable main and diagnostic families; two contrasts per family, 97.5% two-sided paired crossed intervals on new_cells, stable when lower bound positive with >=6/8 positive seeds. R1 and survival descriptive.',
        independence='64 previously unseen maps; distinct recording relative to fitted policies and previous test; same sample, reused frozen weights. Earlier archive study used this recording and electrode selection used it.',
        association='archived signal; no real-time environment feedback')
    with (HERE/'design.json').open('x',encoding='utf8') as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print('locked',digest(spec),len(worlds),'maps',len(electrodes),'electrodes',first,last)
if __name__=='__main__':main()
