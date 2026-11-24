"""Lock all input windows, randomizations and comparisons before the first fit."""
import hashlib
import json
import random
import sys
from itertools import groupby
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ARCHIVE = HERE.parent / 'offline-mea-stim-ablation'
sys.path[:0] = [str(WORK/'.tools/mea_baseline_deps'), str(WORK/'standalone/mea_state2_r2'), str(ARCHIVE)]
from build_inputs import FS, RECORD, read_record, window
from mea_state2_r2.features import digest, fit, transform

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

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
    orig=read(ARCHIVE/'design.json')
    protocol=read(RECORD/'design.json')
    hashes=read(RECORD/'raw-hashes.json')
    checked=read(RECORD/'raw-readback-check.json')
    roi=protocol['roi']
    assert len(roi)==len(set(roi))==322
    warm_a,warm_b,warm_sp,markers=read_record('warmup',hashes,roi)
    assert not markers
    norm=fit([window(warm_sp,warm_a,warm_b,warm_a+(12+i)*FS,-1,0,12+i,roi)['features'] for i in range(24)])
    assert norm==read(ARCHIVE/'sham.json')['normalization']
    train=[];boot=[];test=[];sources={}
    shams=[i for i,v in enumerate(protocol['trials'],1) if v['condition']=='sham']
    assert len(shams)==6
    for pos,i in enumerate(shams):
        name=f'trial_{i:02d}'
        raw=read(RECORD/(name+'-evidence.json')); ch=checked[i-1]
        assert ch['condition']=='sham' and ch['chip']=='22021' and raw['plan']==protocol['trials'][i-1]
        assert raw['pattern_verified'] and raw['intended_edges_present'] and not raw['late_site_rail']
        a,b,sp,events=read_record(name,hashes,roi)
        assert events==raw['frames'] and a>warm_b and raw['metadata']['frame_gaps']==0
        sources[i]=[a,b,events[0]]
        def take(off):
            w=window(sp,a,b,events[0]+off*FS,1000*i+off+20,i,off,roi)
            w['normalized']=list(transform(w['features'],norm))
            return w
        if pos<3:
            off=-7
            for k in range(16):
                train.append(take(off));off+=1
                if len(train)%12==0:
                    boot.append(take(off));off+=1
        else:
            test.extend(take(off) for off in range(-7,13))
    assert (len(train),len(boot),len(test))==(48,4,60)
    for seq in [sorted(train+boot,key=lambda w:w['frame_start']),test]:
        assert all(x['frame_end']<=y['frame_start'] for x,y in zip(seq,seq[1:]))
    assert max(w['frame_end'] for w in train+boot)<test[0]['frame_start']
    series=dict(train=train,bootstrap=boot,holdout=[])
    common=read(ARCHIVE/'sham.json')|dict(series=series,profile='22021_exploration_three_arms',
          state2_source='archived_zero_output',source_sha256=digest(hashes))
    seen=set()
    def integers(value):
        if isinstance(value,int):seen.add(value)
        elif isinstance(value,list):
            for v in value:integers(v)
        elif isinstance(value,dict):
            for v in value.values():integers(v)
    for name in ('design.json','plan.json','protocol.json'):
        for f in WORK.joinpath('output').rglob(name):
            if HERE in f.parents:continue
            o=read(f)
            if isinstance(o,dict):
                for k in ('worlds','holdout_seeds','training_seeds','seeds','episodes','restart_worlds','training_worlds'):
                    if k in o:integers(o[k])
    rng=random.Random(22021111931)
    seeds=rng.sample(range(231000000,232000000),8)
    maps=rng.sample(range(232000000,233000000),48)
    trainmaps=rng.sample(range(233000000,234000000),32)
    assert not set(seeds+maps+trainmaps)&seen
    arms=['masked','ordered','shuffled']
    trainperm={}
    for seed in seeds:
        rr=random.Random(22021111932+seed)
        trainperm[str(seed)]=[j+i for i in range(0,48,12) for j in permute(train[i:i+12],rr)]
    testperm={str(w):permute(test,random.Random(22021111933+w)) for w in maps}
    spec=dict(study='offline_exploration_22021',chip='22021',sample=common['sample'],arms=arms,alpha=1,beta=0,
        feedback_rule='log only: drink at water <=20 followed by net water gain',starting_scene=orig['starting_scene'],
        starting_reserves=orig['starting_reserves'],batch=12,updates=4,eval_decisions=0,training_seeds=seeds,
        episodes={str(v):trainmaps[j*4:j*4+4] for j,v in enumerate(seeds)},worlds=maps,horizon=60,
        context_sha=digest(common),normalization=norm,roi_sha=common['roi_sha'],source_sha=common['source_sha256'],
        calibration='warmup only: seconds [12,36)',train_trials=shams[:3],test_trials=shams[3:],source_frames=sources,
        windows=test,training_order=trainperm,shuffle=testperm,shuffle_unit='2s joint deranged blocks within trial; train also within batch',
        primary='new_cells',secondary=['r1','survived'],contrasts=['ordered minus masked','ordered minus shuffled'],
        stable='each contrast: positive Bonferroni 97.5% interval AND strictly positive in at least 6 of 8 training seeds',
        statistic='40000 paired crossed training-seed and map resamples; descriptive 95% intervals for secondary outcomes',
        bootstrap_seed=22021111934,draws=40000,prior_world_ids_checked=len(seen),
        training_code_sha=hashlib.sha256((HERE/'train.py').read_bytes()).hexdigest(),
        eval_code_sha=hashlib.sha256((HERE/'play60.py').read_bytes()).hexdigest())
    for f,data in [('context.json',common),('design.json',spec)]:
        with (HERE/f).open('x',encoding='utf8') as out:json.dump(data,out,ensure_ascii=False,indent=2)
    print('locked',seeds,shams,len(maps),'contexts',len(train),len(test))
if __name__=='__main__':main()
