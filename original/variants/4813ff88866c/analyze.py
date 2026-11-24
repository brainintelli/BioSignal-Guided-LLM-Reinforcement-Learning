"""Verify every matched receipt. Summarize new maps apart from the earlier discovery study."""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean
import numpy as np
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
def weights(nseed,nworld,draws,seed):
    rng=np.random.default_rng(seed)
    return rng.multinomial(nseed,np.full(nseed,1/nseed),size=draws),rng.multinomial(nworld,np.full(nworld,1/nworld),size=draws)
def effect(m,w,alpha):
    means=m.mean(axis=1);positive=int((means>0).sum());negative=int((means<0).sum())
    sw,ww=w;draw=np.einsum('bi,ij,bj->b',sw,m,ww,optimize=True)/m.size
    lo,hi=np.quantile(draw,[alpha/2,1-alpha/2])
    return dict(delta=float(m.mean()),lo=float(lo),hi=float(hi),positive_seeds=positive,negative_seeds=negative,per_seed=means.tolist(),
        wins=int((m>0).sum()),losses=int((m<0).sum()),ties=int((m==0).sum()),stable=bool(lo>0 and positive>=6))
def main(partial=False):
    s=read(H/'design.json');plan=read(H/'lower-tail-plan.json');cut=plan['cutoff'];old=read(H.parent/'design.json')
    assert plan['design_sha']==sha(H/'design.json') and sha(H.parent/'paired.csv')==plan['prior_pairs_sha']
    assert sha(H/'evaluate.py')==s['runner_sha'] and sha(H.parent/'play60.py')==s['previous_play_sha']
    assert sha(H.parent/'training-audit.json')==s['training_audit_sha'] and digest(old)==s['previous_study_sha']
    assert s['source_sha']!=old['source_sha'] and s['normalization']==old['normalization'] and s['roi_sha']==old['roi_sha']
    seeds=s['training_seeds'];worlds=s['worlds'];arms=list(s['conditions']);wins=s['windows'];pin=digest(s)
    assert len(set(worlds))==64 and len(seeds)==8 and not set(worlds)&set(old['worlds'])
    assert len(wins)==60 and all(a['frame_end']==b['frame_start'] for a,b in zip(wins,wins[1:]))
    out = H/'interim' if partial else H
    out.mkdir(exist_ok=True)
    if partial:
        seeds=[i for i in seeds if all((H/'games'/str(i)/a/'verified-transfer.json').exists() for a in arms)]
        assert 0<len(seeds)<8
    data={}
    for i in seeds:
        for label in arms:
            model,inp=s['conditions'][label];expected=s['states'][f'{i}/{model}']
            assert read(H.parent/'models'/str(i)/model/'update.json')['after']==expected
            folder=H/'games'/str(i)/label
            assert read(folder/'frozen.json')==dict(spec=pin,seed=i,condition=label,state=expected,worlds=64)
            assert {int(f.stem) for f in folder.glob('[0-9]*.json')}==set(worlds)
            for j in worlds:
                obj=read(folder/f'{j}.json');g=obj['game'];rows=g['rows']
                assert obj['spec']==pin and obj['seed']==i and obj['condition']==label and obj['frozen']==expected
                assert g['world']==j and g['mode']==inp and len(rows)==g['decisions'] and rows[0]['tick']==0
                assert g['end_tick']==rows[-1]['end_tick']<=60 and (g['end_tick']==60 or not g['survived']) and bool(g['survived'])==(g['health']>0)
                assert math.isclose(sum(r['reward'] for r in rows),g['r1'],abs_tol=1.e-12)
                order=s['shuffle'][str(j)] if inp=='shuffled' else list(range(60))
                assert sorted(order)==list(range(60)) and all(k//20==v//20 for k,v in enumerate(order))
                tick=0;cells=0
                for r in rows:
                    assert r['tick']==tick and tick<r['end_tick'] and cells<=r['new_cells_at_end']
                    w=wins[order[tick]];vec=([0.]*4 if inp=='masked' else w['normalized'])
                    assert r['features']==vec and (r['window'],r['events'])==(w['index'],w['events_sha'])
                    assert r['r2_contribution']==0 and (not r['cut'] or r['end_tick']==60)
                    assert len(r['transition_sha'])==64
                    tick=r['end_tick'];cells=r['new_cells_at_end']
                assert cells==g['new_cells']
                data[i,j,label]=dict(seed=i,world=j,condition=label,training=model,input=inp,new_cells=cells,low=int(cells<=cut),
                    r1=g['r1'],survived=g['survived'],end_tick=tick,decisions=len(rows))
    keys=('new_cells','low','r1','survived')
    groups={a:{k:mean(v[k] for v in data.values() if v['condition']==a) for k in keys} for a in arms}
    pairs=[]
    for i in seeds:
        for j in worlds:
            r=dict(seed=i,world=j)
            for a in arms:
                for k in keys:r[a+'_'+k]=data[i,j,a][k]
            pairs.append(r)
    csvwrite(out/'episodes.csv',list(data.values()));csvwrite(out/'paired.csv',pairs)
    assert len(data)==len(seeds)*64*5 and len(pairs)==len(seeds)*64
    if not partial:assert len(data)==2560 and len(pairs)==512
    wt=weights(len(seeds),64,s['draws'],s['resampling_seed']);effects={}
    for a in s['main']+s['mechanism']:
        effects[a]={}
        for k in ('new_cells','r1','survived'):
            m=np.array([[data[i,j,'ordered'][k]-data[i,j,a][k] for j in worlds] for i in seeds])
            effects[a][k]=effect(m,wt,.025 if k=='new_cells' else .05)
    m=np.array([[data[i,j,'masked']['low']-data[i,j,'ordered']['low'] for j in worlds] for i in seeds])
    low=effect(m,wt,.05)
    original=list(csv.DictReader((H.parent/'paired.csv').open(encoding='utf-8-sig')))
    oldworld=old['worlds'];lookup={(int(r['seed']),int(r['world'])):r for r in original}
    oldmask=np.array([[int(lookup[i,j]['masked_new_cells'])<=cut for j in oldworld] for i in old['training_seeds']],dtype=int)
    oldtrue=np.array([[int(lookup[i,j]['ordered_new_cells'])<=cut for j in oldworld] for i in old['training_seeds']],dtype=int)
    low_original=effect(oldmask-oldtrue,weights(8,48,s['draws'],s['resampling_seed']+1),.05)
    lows=dict(cutoff=cut,original_masked=int(oldmask.sum()),original_ordered=int(oldtrue.sum()),original=low_original,new=low)
    res=dict(partial=partial,seeds_evaluated=seeds,design_sha=pin,games=len(data),decisions=sum(v['decisions'] for v in data.values()),pairs=len(pairs),groups=groups,effects=effects,lower_tail=lows)
    (out/'comparison.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(dict(groups=groups,main={k:effects[k]['new_cells'] for k in effects},low=lows,decisions=res['decisions'])))
if __name__=='__main__':main()
