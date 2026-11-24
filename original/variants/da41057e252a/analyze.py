"""Verify all matched game receipts and estimate effects for the three registered endpoints."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,median

import numpy as np

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
def read(path):return json.loads(path.read_text(encoding='utf8'))
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def collect(s):
    d= digest(s);train=read(TRAIN/'design.json');mix=train['reward_mix'];audit={};data={}
    assert len(s['worlds'])==len(set(s['worlds']))==48 and s['seeds']==train['training_seeds'] and s['arms']==['r1','ordered']
    assert len(s['windows'])==s['horizon']==60 and s['training_max_ratio'] and max(s['training_max_ratio'].values())<=.05+1e-15
    for seed in s['seeds']:
        for arm in s['arms']:
            model=TRAIN/'models'/str(seed)/arm
            expected=read(model/'update.json')['after']
            assert hashlib.sha256((model/'checkpoint/manifest.json').read_bytes()).hexdigest()==s['checkpoints'][f'{seed}/{arm}']
            folder=HERE/'runs'/str(seed)/arm
            pin=read(folder/'frozen.json')
            assert pin==dict(seed=seed,arm=arm,state=expected,worlds=48,design_sha=d)
            assert {int(f.stem) for f in folder.glob('[0-9]*.json')}==set(s['worlds'])
            n=0;ma=0.;mass=native_mass=0;labels={-1:0,0:0,1:0}
            for world in s['worlds']:
                rec=read(folder/f'{world}.json');g=rec['game'];rows=g['rows']
                assert rec['frozen']==expected and rec['design_sha']==d and rec['seed']==seed
                assert g['world']==world and g['mode']==arm and len(rows)==g['decisions'] and rows[0]['tick']==0
                assert rows[-1]['end_tick']==g['end_tick']<=60 and bool(g['survived'])==(g['health']>0)
                assert not g['survived'] or g['end_tick']==60
                assert all(a['end_tick']==b['tick'] and not a['cut'] for a,b in zip(rows,rows[1:]))
                assert math.isclose(sum(r['reward'] for r in rows),g['r1'],abs_tol=1e-12)
                verified=[]
                for i,r in enumerate(rows):
                    n+=1;tick=r['tick'];dt=r['end_tick']-tick
                    assert dt>0 and len(r['transition_sha'])==64
                    win=s['windows'][tick]
                    assert (r['window'],r['features'],r['events'])==(win['index'],win['normalized'],win['events_sha'])
                    drank=r['action']['verb']=='drink' and r['water_after']>r['water_before']
                    assert drank==r['drink_success']
                    if drank:verified.append((tick,r['end_tick']))
                    native=r['reward'];label=r['feedback_label'];stop=not g['survived'] and i==len(rows)-1
                    allowed=((i+1)%s['stride']==0 and not stop and not r['cut'])
                    assert (abs(label)==1 if allowed else label==0)
                    scaled=label*mix['scale_fraction']*min(abs(native),mix['reference_per_tick']*sum(mix['gamma_tick']**j for j in range(dt)))
                    part=scaled*(train['beta'] if arm=='ordered' else 0.)
                    assert math.isclose(scaled,r['r2_scaled'],abs_tol=1e-15) and math.isclose(part,r['contribution'],abs_tol=1e-15)
                    assert abs(part)<=.05*abs(native)+1e-15 and math.isclose(r['total_reward'],native+part,abs_tol=1e-15)
                    labels[label]+=1;mass+=abs(part);native_mass+=abs(native)
                    if native:ma=max(ma,abs(part/native))
                first=verified[0] if verified else (None,None)
                assert (g['first_drink_start'],g['first_drink_confirmed'])==first and g['wait_to_drink_60']==(60 if first[0] is None else first[0])
                data[seed,world,arm]=dict(seed=seed,world=world,arm=arm,survived=g['survived'],end_tick=g['end_tick'],first_drink_start=first[0],
                  first_drink_confirmed=first[1],drank=int(bool(verified)),wait_to_drink_60=g['wait_to_drink_60'],decisions=len(rows),r1=g['r1'])
            audit[f'{seed}/{arm}']=dict(decisions=n,max_r2_fraction=ma,absolute_contribution=mass,absolute_r1=native_mass,labels=labels)
    assert len(data)==768
    write_csv(HERE/'episodes.csv',list(data.values()))
    return data,audit

def describe(rows):
    drink=[r['first_drink_start'] for r in rows if r['drank']]
    return dict(n=len(rows),alive=sum(r['survived'] for r in rows),drank=len(drink),deaths=len(rows)-sum(r['survived'] for r in rows),
                survived=mean(r['survived'] for r in rows),end_tick=mean(r['end_tick'] for r in rows),
                wait_to_drink_60=mean(r['wait_to_drink_60'] for r in rows),
                actual_drink_mean=mean(drink) if drink else None,actual_drink_median=median(drink) if drink else None,
                alive_by_tick=[sum(r['end_tick']>t or r['survived'] for r in rows)/len(rows) for t in range(61)],
                first_drink_by_tick=[sum(t0<=t for t0 in drink)/len(rows) for t in range(61)])

def main():
    s=read(HERE/'design.json');data,audit=collect(s);seeds=s['seeds'];worlds=s['worlds'];arms=s['arms'];keys=s['endpoints']
    groups={a:describe([r for r in data.values() if r['arm']==a]) for a in arms}
    paired=[]
    for seed in seeds:
        for w in worlds:
            row=dict(seed=seed,world=w)
            for arm in arms:
                g=data[seed,w,arm];row.update({arm+'_'+k:g[k] for k in [*keys,'drank','first_drink_start','first_drink_confirmed']})
            paired.append(row)
    write_csv(HERE/'paired.csv',paired)
    rng=np.random.default_rng(s['seed_bootstrap']);n=s['draws']
    sw=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=n)
    ww=rng.multinomial(len(worlds),np.full(len(worlds),1/len(worlds)),size=n)
    effects={}
    for k in keys:
        mat=np.array([[data[i,j,'ordered'][k]-data[i,j,'r1'][k] for j in worlds] for i in seeds])
        draws=np.einsum('bi,ij,bj->b',sw,mat,ww,optimize=True)/(len(seeds)*len(worlds))
        tail=(1-s['ci_each'])/2;lo,hi=np.quantile(draws,[tail,1-tail]);direction=-1 if k=='wait_to_drink_60' else 1
        seed_effects=mat.mean(axis=1);fav=int((seed_effects*direction>0).sum())
        effects[k]=dict(delta=float(mat.mean()),interval=[float(lo),float(hi)],per_seed=seed_effects.tolist(),favorable_seeds=fav,
            favorable_pairs=int((mat*direction>0).sum()),unfavorable_pairs=int((mat*direction<0).sum()),ties=int((mat==0).sum()),
            stable=bool((hi<0 if direction==-1 else lo>0) and fav>=6))
    both=[r['ordered_first_drink_start']-r['r1_first_drink_start'] for r in paired if r['ordered_drank'] and r['r1_drank']]
    diagnostics=dict(both_drank=len(both),both_drink_time_diff=mean(both) if both else None,
         only_ordered=sum(r['ordered_drank'] and not r['r1_drank'] for r in paired),
         only_control=sum(r['r1_drank'] and not r['ordered_drank'] for r in paired),neither=sum(not r['ordered_drank'] and not r['r1_drank'] for r in paired))
    res=dict(design_sha=digest(s),games=len(data),action_decisions=sum(v['decisions'] for v in audit.values()),groups=groups,
        effects=effects,stable_all=all(v['stable'] for v in effects.values()),paired_drinks=diagnostics,reward_audit=audit)
    (HERE/'comparison.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(games=len(data),actions=res['action_decisions'],groups={k:{x:v for x,v in g.items() if not isinstance(v,list)} for k,g in groups.items()},effects=effects,paired_drinks=diagnostics)))
if __name__=='__main__':main()
