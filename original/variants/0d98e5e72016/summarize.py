"""Audit all played episodes and estimate paired contrasts across common seeds and maps."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent.parent
OUTCOMES = ('survived', 'end_tick', 'r1', 'new_cells')
ORDERED = 'trained_ordered'
MASKED = 'control_masked'
SHUFFLED = 'trained_shuffled'

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def collect(spec):
    archive = HERE.parent
    pilot=read(archive/'severe/design.json')
    assert sha(archive/'severe/design.json')==spec['parent_pilot_design_sha256']
    assert sha(archive/'freeze.py')==spec['frozen_runner_sha256']
    assert not set(spec['worlds']) & set(pilot['worlds'])
    assert spec['source_sha']==sha(STUDY.parent/spec['source_file'])
    for key in ('aligned_electrodes','remap','windows','horizon','scenes','normalization','roi_sha','source_frames','modes'):
        assert spec[key]==pilot[key]
    assert spec['seeds']==read(STUDY/'design.json')['training_seeds']
    canonical=json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)
    design_hash=hashlib.sha256(canonical.encode()).hexdigest()
    seeds,worlds,modes=(spec[k] for k in ('seeds','worlds','modes'))
    assert (len(set(seeds)),len(set(worlds)),len(modes))==(8,48,3)
    assert len(set(spec['aligned_electrodes']))==322 and (spec['alpha'],spec['beta'])==(1,0)
    assert len(list((HERE/'runs').glob('*/*.json')))==len(seeds)*len(modes)
    data,episodes,checksums={},{},{}
    transitions={}
    for seed in seeds:
        for mode in modes:
            path=HERE/'runs'/str(seed)/(mode+'.json')
            record=read(path)
            group=STUDY/'models'/str(seed)/'masked' if mode==MASKED else STUDY/'pre-stim-baseline/models'/str(seed)/'baseline'
            assert record['frozen']==read(group/'update.json')['after']
            assert (record['study_sha'],record['seed'],record['mode'])==(design_hash,seed,mode)
            assert [g['world'] for g in record['games']]==worlds
            checksums[path.relative_to(HERE).as_posix()]=sha(path)
            for game in record['games']:
                world=game['world']
                assert game['mode']==mode and game['survived'] in (0,1)
                assert bool(game['survived'])==(game['health']>0)
                assert game['end_tick']<=60 and (game['end_tick']==60 or not game['survived'])
                rows=game['rows']
                assert rows and len(rows)==game['decisions'] and rows[0]['tick']==0 and rows[-1]['end_tick']==game['end_tick']
                assert all(a['end_tick']==b['tick'] and not a['cut'] for a,b in zip(rows,rows[1:]))
                assert math.isclose(sum(r['reward'] for r in rows),game['r1'],abs_tol=1e-12)
                ticks=[r['tick'] for r in rows]
                assert len(set(ticks))==len(ticks)
                order=spec['shuffle'][str(world)] if mode==SHUFFLED else list(range(60))
                assert sorted(order)==list(range(60))
                for row in rows:
                    assert row['tick']<row['end_tick'] and len(row['transition_sha'])==64
                    window=spec['windows'][order[row['tick']]]
                    vec=[0.]*4 if mode==MASKED else window['normalized']
                    assert (row['events'],row['window'],row['features'])==(window['events_sha'],window['index'],vec)
                drinks=[r['tick'] for r in rows if r['action']['verb']=='drink']
                key=(mode,seed,world)
                assert key not in data
                data[key]={k:game[k] for k in OUTCOMES}
                episodes[key]={'mode':mode,'training_seed':seed,'world':world,**data[key], 'decisions':len(rows),'end_health':game['health'],'end_water':game['thirst'],'first_drink_start_tick':drinks[0] if drinks else None}
                transitions[key]=len(rows)
    provenance = HERE/'shards/constituents.json'
    if provenance.exists():
        fragments=read(provenance)
        assert len(fragments)==9
        last=seeds[-1]
        for mode in modes:
            combined=read(HERE/'runs'/str(last)/(mode+'.json'))
            rebuilt=[]
            for i in range(3):
                shard=HERE/'shards'/str(i)
                sub=read(shard/'design.json')
                assert sub==spec|{'worlds':worlds[16*i:16*(i+1)]}
                piece=shard/'runs'/str(last)/(mode+'.json')
                assert sha(piece)==fragments[piece.relative_to(HERE).as_posix()]
                item=read(piece)
                assert item['frozen']==combined['frozen'] and item['mode']==mode and item['seed']==last
                rebuilt.extend(item['games'])
            assert rebuilt==combined['games']
    assert len(episodes)==1152
    write_csv(HERE/'episodes.csv',list(episodes.values()))
    drinking={}
    for mode in modes:
        arm=[e for e in episodes.values() if e['mode']==mode]
        first=[e['first_drink_start_tick'] for e in arm if e['first_drink_start_tick'] is not None]
        drinking[mode]={'drank_and_survived':sum(e['first_drink_start_tick'] is not None and e['survived']==1 for e in arm),
                        'no_drink_and_died_at_33':sum(e['first_drink_start_tick'] is None and e['survived']==0 and e['end_tick']==33 for e in arm),
                        'first_drink_min':min(first),'first_drink_max':max(first)}
    return data,checksums,sum(transitions.values()),design_hash,drinking

def analyze(spec, data):
    seeds,worlds,modes=(spec[k] for k in ('seeds','worlds','modes'))
    means={}
    for mode in modes:
        g=[row for (m,s,w),row in data.items() if m==mode]
        means[mode]={'n':len(g),'alive':sum(row['survived'] for row in g),**{k:float(np.mean([row[k] for row in g])) for k in OUTCOMES}}
    rng=np.random.default_rng(spec['bootstrap']['seed'])
    bs=spec['bootstrap']['draws']
    # Resample each crossed factor once per draw, preserving matches across conditions.
    # Multinomial counts implement sampling cluster indices with replacement.
    sw=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=bs)
    ww=rng.multinomial(len(worlds),np.full(len(worlds),1/len(worlds)),size=bs)
    paired=[]
    effects={}
    for arm in (MASKED,SHUFFLED):
        by_metric={}
        for k in OUTCOMES:
            d=np.array([[data[(ORDERED,s,w)][k]-data[(arm,s,w)][k] for w in worlds] for s in seeds],dtype=float)
            bootstrap=(np.einsum('bi,ij,bj->b',sw,d,ww,optimize=True)/(len(seeds)*len(worlds)))
            lo,hi=np.quantile(bootstrap,[.025,.975])
            by_metric[k]={'mean':float(d.mean()), 'ci95':[float(lo),float(hi)],'higher':int((d>0).sum()),'lower':int((d<0).sum()),'ties':int((d==0).sum()),
                'positive_seeds':int((d.mean(axis=1)>0).sum()),'positive_worlds':int((d.mean(axis=0)>0).sum()),
                'by_seed':[float(v) for v in d.mean(axis=1)],'by_world':[float(v) for v in d.mean(axis=0)]}
        effects[arm]=by_metric
    for s in seeds:
        for w in worlds:
            row={'training_seed':s,'world':w}
            for m in modes:
                for k in OUTCOMES:row[m+'_'+k]=data[m,s,w][k]
            paired.append(row)
    write_csv(HERE/'paired.csv', paired)
    return means,effects

def main():
    spec=read(HERE/'design.json')
    data,files,n_action,digest,drinking=collect(spec)
    means,effects=analyze(spec,data)
    result={'episodes':len(data),'decisions':n_action,'distinct_organoids':1,'modes':means,'effects_ordered_minus':effects,'design_digest':digest,'raw_file_sha256':files,'posthoc_drinking_check':drinking}
    (HERE/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'n':len(data),'decisions':n_action,'means':means,'effects':{m:{k:(v['mean'],v['ci95'],v['higher'],v['lower'],v['ties']) for k,v in d.items()} for m,d in effects.items()}}))
if __name__=='__main__': main()
