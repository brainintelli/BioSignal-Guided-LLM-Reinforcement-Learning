"""Form matched replay inputs only from verified and recovered 1 Hz and zero-output trials."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]; RUN=OUT/'single-site-v1'
OLD=ROOT/'output/mea_predictability_22021_20261118T093720Z/offline-mea-stim-ablation'
PREV=ROOT/'output/mea_predictability_22021_20261119T042416Z/offline-five-arm-replication'
os.environ['HDF5_PLUGIN_PATH']=str(ROOT/'.tools/mea_maxwell_hdf5_plugin')
sys.path[:0]=[str(OLD),str(ROOT/'.tools/mea_baseline_deps'),str(ROOT/'standalone/mea_state2_r2'),str(ROOT/'standalone/mea_stimulation_v2')]
import build_inputs
from build_inputs import read,window,read_record,write,np
from mea_state2_r2.features import digest,fit,transform
from mea_stim_v2.recording import H5Recording
from mea_stim_v2.qc import stability

def verify_and_register():
    binding=read(OUT/'baseline-binding.json'); cfg=SimpleNamespace(**binding)
    if not (RUN/'neutral-check.json').is_file():return 'No qualified routed neutral reference'
    neutral=read(RUN/'neutral-check.json');roi=neutral['roi'];site=read(RUN/'sites.json');chs=site['channels']
    manifest=read(OUT/'verified-recordings.json');prefix='frequency-retest/single-site-v1/'
    hashes={k[len(prefix):]:v for k,v in manifest.items() if k.startswith(prefix) and k.endswith('.h5')}
    build_inputs.RECORD=RUN
    trials=read(RUN/'schedule.json');end=read(RUN/'end.json')
    nf,nl,sp,events=read_record('neutral',hashes,roi)
    assert len(events)==3
    cal=[window(sp,nf,nl,events[-1]+off*20000,off,0,off,roi) for off in range(8,32)]
    norm=fit([w['features'] for w in cal])
    rows=[];series={c:{s:[] for s in ('train','holdout','bootstrap')} for c in ('one_hz','zero')}
    for tr in trials:
        name='trial_%02d'%tr['index']
        fp=RUN/(name+'-evidence.json')
        if not fp.exists():continue
        raw=read(fp);f,l,sp,ev=read_record(name,hashes,roi)
        assert raw['row']==tr and ev==[e['frame'] for e in raw['events']]
        active=tr['condition']=='one_hz'
        assert [v-ev[0] for v in ev]==([0,20000,40000] if active else [0])
        with H5Recording(RUN/name/(name+'.raw.h5'),cfg) as r:
            assert r.metadata['chip_id']==binding['chip_id'] and not r.metadata['frame_gaps']
            assert all(dict(r.mapping)[c]==e for c,e in zip(chs,site['electrodes']))
            t=r.events[0]['time_s']
            rel=r.sample(chs,t-.004,2.71);rails=(rel==0)|(rel==1023)
            ref=np.median(r.sample(chs,t-3),axis=1)
            peak=float(np.max(np.abs(rel.astype(float)-ref[:,None]))*r.metadata['lsb_uv'])
            observed=[];late=[]
            for k in range(3):
                idx=80+20000*k
                observed.append(np.flatnonzero(~rails[:,idx-8:idx].any(axis=1)&rails[:,idx:idx+40].any(axis=1)).tolist())
                late.append(rails[:,idx+12920:idx+13920].sum(axis=1).tolist())
            assert observed==raw['onsets'] and late==raw['late_counts']
            tail=r.sample(chs,t+14)
            stable=stability(r.sample(roi,t-3),r.sample(roi,t+14))
            clean=observed==([[0]]*3 if active else [[],[],[]]) and not np.any(late) and not np.any((tail==0)|(tail==1023)) and (active or peak<1000)
            admitted=bool(clean and stable['passed'])
            assert clean==raw['clean'] and admitted==raw['state2_admitted']
            rows.append(dict(row=tr,observed=observed,late=late,clean=bool(clean),recovery=stable['fraction_channels_passed'],admitted=admitted))
        if admitted:
            for off in (range(4,17) if tr['block']<=4 else [4,9,14]):
                key='holdout' if tr['block']>4 else 'bootstrap' if off==16 else 'train'
                w=window(sp,f,l,ev[0]+off*20000,tr['index']*100+off,tr['index'],off,roi)
                w['normalized']=list(transform(w['features'],norm))
                series[tr['condition']][key].append(w)
    decision=dict(trials=rows,eligible_by_condition={c:{k:len(v) for k,v in s.items()} for c,s in series.items()})
    all_ok=(end['completed']==len(trials)==12 and end['closed'] and not end['cleanup'] and
            all(v=={'train':48,'holdout':6,'bootstrap':4} for v in decision['eligible_by_condition'].values()))
    decision['eligible_for_matched_training']=all_ok;write(OUT/'admission.json',decision)
    if not all_ok:return 'Insufficient matched clean train/holdout blocks'
    directory=OUT/'offline-models';directory.mkdir(exist_ok=True)
    source=digest(hashes);ordered=series['one_hz'];shuffled={}
    for j,(phase,s) in enumerate(ordered.items()):
        permutation=list(range(len(s)));rng=random.Random(277011120+j)
        while True:
            rng.shuffle(permutation)
            if all(k!=v for k,v in enumerate(permutation)):break
        shuffled[phase]=[s[i] for i in permutation]
    spec=read(PREV/'design.json');arms=['masked','ordered','shuffled','sham']
    contexts={}
    for arm,s in zip(arms,[series['zero'],ordered,shuffled,series['zero']]):
        ctx=dict(chip=binding['chip_id'],sample=binding['organoid_id'],clock='MaxOne 20 kHz frames',normalization=norm,
                 source_sha256=source,roi_sha=digest(roi),profile='27701_1Hz_single_site_matched_replay',state2_source=arm,series=s)
        contexts[arm]=digest(ctx);write(directory/(arm+'.json'),ctx)
    spec.update(chip='27701',session_id=binding['session_id'],organoid_id=binding['organoid_id'],contexts=contexts,
                arms=arms,source_sha=source,contrasts=[['ordered','masked'],['ordered','shuffled'],['ordered','sham']],
                baseline_windows=cal,selection='1 Hz, six matched physical blocks',supplemental_clock=6)
    write(directory/'design.json',spec)
    return 'Registered matched inputs'
if __name__=='__main__':print(verify_and_register())
