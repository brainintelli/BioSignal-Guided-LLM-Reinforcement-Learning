"""Audit frozen receipts and calculate the registered matched-map differences."""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dumpcsv(file,rows):
    with (HERE/file).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)

def main():
    s=read(HERE/'design.json');c=read(HERE/'context.json')
    audits=read(HERE/'training-audit.json');au={(a['seed'],a['arm']):a for a in audits}
    seeds=s['training_seeds'];worlds=s['worlds'];arms=s['arms'];pin=digest(s)
    assert len(au)==24 and len(set(seeds))==8 and len(set(worlds))==48 and s['primary']=='new_cells'
    assert c['chip']=='22021' and digest(c)==s['context_sha'] and s['alpha']==1 and s['beta']==0
    for name,k in [('train.py','training_code_sha'),('play60.py','eval_code_sha')]:
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==s[k]
    train_ids=set(s['train_trials']);test_ids=set(s['test_trials']);assert not train_ids & test_ids
    wins=s['windows'];assert len(wins)==len({v['index'] for v in wins})==60 and set(w['trial'] for w in wins)==test_ids
    assert set(w['trial'] for w in c['series']['train'])==train_ids
    assert max(w['frame_end'] for k in ('train','bootstrap') for w in c['series'][k])<=min(w['frame_start'] for w in wins)
    data={};metrics=['new_cells','r1','survived']
    for seed in seeds:
        for arm in arms:
            model=HERE/'models'/str(seed)/arm
            up=read(model/'update.json');audit=au[seed,arm]
            assert audit['after']==up['after'] and audit['steps']==48 and audit['reward_contribution']==0
            assert hashlib.sha256((model/'checkpoint/manifest.json').read_bytes()).hexdigest()==audit['checkpoint_manifest']
            folder=HERE/'games'/str(seed)/arm
            assert read(folder/'frozen.json')==dict(seed=seed,arm=arm,worlds=48,state=up['after'],spec=pin)
            assert {int(f.stem) for f in folder.glob('[0-9]*.json')}==set(worlds)
            for world in worlds:
                d=read(folder/f'{world}.json');g=d['game'];rows=g['rows']
                assert d['frozen']==up['after'] and d['seed']==seed and d['spec']==pin and g['mode']==arm and g['world']==world
                assert len(rows)==g['decisions'] and rows[0]['tick']==0 and rows[-1]['end_tick']==g['end_tick']<=60
                assert bool(g['survived'])==(g['health']>0) and (not g['survived'] or g['end_tick']==60)
                assert g['new_cells']==rows[-1]['new_cells_at_end']>=0 and math.isclose(sum(r['reward'] for r in rows),g['r1'],abs_tol=1e-12)
                order=s['shuffle'][str(world)] if arm=='shuffled' else list(range(60))
                assert sorted(order)==list(range(60))
                assert all(wins[i]['trial']==wins[j]['trial'] for i,j in enumerate(order))
                last=0;visited=0
                for r in rows:
                    w=wins[order[r['tick']]]
                    assert r['tick']==last and r['end_tick']>last and r['new_cells_at_end']>=visited
                    assert r['r2_contribution']==0 and r['r2_logged'] in (0,1)
                    assert (r['window'],r['events'])==(w['index'],w['events_sha'])
                    assert r['features']==([0.]*4 if arm=='masked' else w['normalized'])
                    assert len(r['transition_sha'])==64 and (not r['cut'] or r['end_tick']==60)
                    last=r['end_tick'];visited=r['new_cells_at_end']
                data[seed,world,arm]=dict(seed=seed,world=world,arm=arm,new_cells=g['new_cells'],r1=g['r1'],
                    survived=g['survived'],end_tick=g['end_tick'],decisions=len(rows),r2_logged=sum(r['r2_logged'] for r in rows))
    groups={a:{k:mean(x[k] for x in data.values() if x['arm']==a) for k in metrics} for a in arms}
    paired=[]
    for seed in seeds:
        for world in worlds:
            r=dict(seed=seed,world=world)
            for a in arms:
                for k in metrics:r[a+'_'+k]=data[seed,world,a][k]
            paired.append(r)
    assert len(data)==1152 and len(paired)==384
    dumpcsv('episodes.csv',list(data.values()));dumpcsv('paired.csv',paired)
    rng=np.random.default_rng(s['bootstrap_seed'])
    sw=rng.multinomial(8,[1/8]*8,size=s['draws']);ww=rng.multinomial(48,[1/48]*48,size=s['draws'])
    effects={}
    for a in ('masked','shuffled'):
        effects[a]={}
        for k in metrics:
            m=np.array([[data[i,j,'ordered'][k]-data[i,j,a][k] for j in worlds] for i in seeds])
            draws=np.einsum('bi,ij,bj->b',sw,m,ww,optimize=True)/m.size
            alpha=.025 if k=='new_cells' else .05
            lo,hi=np.quantile(draws,[alpha/2,1-alpha/2]);per=m.mean(axis=1);positive=int((per>0).sum())
            effects[a][k]=dict(delta=float(m.mean()),lo=float(lo),hi=float(hi),positive_seeds=positive,
                per_seed=per.tolist(),pair_wins=int((m>0).sum()),pair_losses=int((m<0).sum()),pair_ties=int((m==0).sum()),
                stable=bool(lo>0 and positive>=6) if k=='new_cells' else None)
    res=dict(games=len(data),actions=sum(x['decisions'] for x in data.values()),groups=groups,effects=effects,design_sha=pin)
    (HERE/'comparison.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(res))
if __name__=='__main__':main()
