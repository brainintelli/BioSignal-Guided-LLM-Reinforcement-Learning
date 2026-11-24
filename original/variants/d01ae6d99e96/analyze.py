"""Validate all step receipts, reconstruct eating and drinking, and compare paired conditions."""
from pathlib import Path
import csv,hashlib,json,math
from statistics import mean,median
import numpy as np
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def csvwrite(name,rows):
    with (H/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
def diff(m,sw,ww,alpha,direction):
    by_seed=m.mean(axis=1);x=np.einsum('bi,ij,bj->b',sw,m,ww,optimize=True)/m.size
    lo,hi=np.quantile(x,[alpha/2,1-alpha/2]);n=int((by_seed*direction>0).sum());bad=int((by_seed*direction<0).sum())
    return dict(delta=float(m.mean()),lo=float(lo),hi=float(hi),per_seed=by_seed.tolist(),
        favorable_seeds=n,adverse_seeds=bad,stable_gain=bool(n>=6 and (lo>0 if direction==1 else hi<0)),
        stable_harm=bool(bad>=6 and (hi<0 if direction==1 else lo>0)),wins=int((m*direction>0).sum()),
        losses=int((m*direction<0).sum()),ties=int((m==0).sum()),level=1-alpha)
def main():
    s=read(H/'design.json');orig=read(H.parent/'design.json');pilot=read(H/'scene-check.json')
    assert sha(H/'scene-check.json')==s['pilot_sha'] and pilot['pins']==s['game_pins']
    assert digest(orig)==s['training_design_sha'] and sha(H.parent/'play60.py')==s['play_sha'] and sha(H/'evaluate.py')==s['runner_sha']
    assert sha(H.parent/'training-audit.json')==s['training_audit_sha'] and s['alpha']==1 and s['beta']==0
    prev=read(H.parent/'baseline-replication-64/design.json')
    assert digest(prev)==s['previous_eval_design_sha'] and prev['windows']==s['windows']
    seeds=s['training_seeds'];worlds=s['worlds'];arms=s['arms'];scenes=s['scenes'];pin=digest(s)
    assert len(set(worlds))==64 and not set(worlds)&(set(prev['worlds'])|set(orig['worlds'])|set(pilot['worlds']))
    data={}
    for i in seeds:
        for scene in scenes:
            for arm in arms:
                base=H/'games'/str(i)/scene/arm;state=s['states'][f'{i}/{arm}']
                assert read(H.parent/'models'/str(i)/arm/'update.json')['after']==state
                assert read(base/'frozen.json')==dict(scene=scene,seed=i,arm=arm,state=state,worlds=64,spec=pin)
                assert {int(f.stem) for f in base.glob('[0-9]*.json')}==set(worlds)
                for j in worlds:
                    d=read(base/f'{j}.json');g=d['game'];rows=g['rows'];initial=scenes[scene]
                    assert d['spec']==pin and d['seed']==i and d['scene']==scene and d['frozen']==state
                    assert g['world']==j and g['mode']==arm and len(rows)==g['decisions']
                    assert bool(g['survived'])==(g['health']>0) and (g['end_tick']==60 or not g['survived'])
                    assert math.isclose(sum(x['reward'] for x in rows),g['r1'],abs_tol=1e-12)
                    order=s['shuffle'][str(j)] if arm=='shuffled' else list(range(60))
                    assert sorted(order)==list(range(60)) and all(k//20==v//20 for k,v in enumerate(order))
                    tick=cells=0;water=initial['thirst'];food=initial['hunger'];first={'drink':None,'eat':None}
                    for r in rows:
                        e=r['resource_event']
                        assert (r['tick'],r['end_tick'],r['transition_sha'])==(e['start'],e['end'],e['tr_sha'])
                        assert tick==r['tick']<r['end_tick']<=60 and cells<=r['new_cells_at_end']
                        assert math.isclose(water,e['water0'],abs_tol=1e-9) and math.isclose(food,e['food0'],abs_tol=1e-9)
                        w=s['windows'][order[tick]]
                        assert (r['events'],r['window'])==(w['events_sha'],w['index'])
                        assert r['features']==([0.]*4 if arm=='masked' else w['normalized'])
                        for verb,pre,post in [('drink','water0','water1'),('eat','food0','food1')]:
                            success=r['action']['verb']==verb and e[post]>e[pre]
                            assert success==(e[verb] is not None)
                            if success:
                                assert tick<e[verb]<=e['end'] and e[verb]==tick+1
                                if first[verb] is None:first[verb]=e[verb]
                        assert r['r2_contribution']==0 and r['r2_logged']==int(not r['cut'] and r['action']['verb']=='drink' and e['water0']<=20 and e['water1']>e['water0'])
                        tick=r['end_tick'];cells=r['new_cells_at_end'];water=e['water1'];food=e['food1']
                    assert first==g['first_success_tick'] and tick==g['end_tick'] and cells==g['new_cells']
                    wait={k:60 if v is None else v for k,v in first.items()};assert wait==g['restricted_wait']
                    data[scene,i,j,arm]=dict(scene=scene,seed=i,world=j,arm=arm,survived=g['survived'],end_tick=tick,
                        drink_tick=first['drink'],eat_tick=first['eat'],drank=int(first['drink'] is not None),
                        ate=int(first['eat'] is not None),drink_wait=wait['drink'],eat_wait=wait['eat'],cells=cells,
                        decisions=len(rows),r1=g['r1'],r2_logged=sum(r['r2_logged'] for r in rows))
    assert len(data)==4608
    metrics=['survived','drink_wait','eat_wait','cells','end_tick','drank','ate']
    csvwrite('episodes.csv',list(data.values()))
    pairs=[];groups={};effects={}
    rng=np.random.default_rng(s['bootstrap_seed'])
    sw=rng.multinomial(8,[1/8]*8,size=s['draws']);ww=rng.multinomial(64,[1/64]*64,size=s['draws'])
    for scene in scenes:
        groups[scene]={};effects[scene]={}
        for arm in arms:
            rows=[data[scene,i,j,arm] for i in seeds for j in worlds]
            g={k:mean(r[k] for r in rows) for k in metrics}
            for verb in ['drink','eat']:
                observed=[r[verb+'_tick'] for r in rows if r[verb+'_tick'] is not None]
                g[verb+'_confirmed_median']=median(observed) if observed else None
                g[verb+'_count']=len(observed)
            if scene in ('thirst','hunger'):
                key='drank' if scene=='thirst' else 'ate'
                g['resource_and_survival']={f'{event}_{alive}':sum(r[key]==event and r['survived']==alive for r in rows)
                    for event in (0,1) for alive in (0,1)}
            g['alive']=sum(r['survived'] for r in rows);g['n']=len(rows);groups[scene][arm]=g
        for i in seeds:
            for j in worlds:
                pair=dict(scene=scene,seed=i,world=j)
                for arm in arms:
                    for k in metrics+['drink_tick','eat_tick']:pair[arm+'_'+k]=data[scene,i,j,arm][k]
                pairs.append(pair)
        for a in ('masked','shuffled'):
            effects[scene][a]={}
            for k in metrics:
                mat=np.array([[data[scene,i,j,'ordered'][k]-data[scene,i,j,a][k] for j in worlds] for i in seeds])
                target=(scene=='thirst' and k=='drink_wait') or (scene=='hunger' and k=='eat_wait')
                level=.05/6 if k=='survived' else .05/4 if target else .05
                e=diff(mat,sw,ww,level,-1 if k in ('drink_wait','eat_wait') else 1)
                e['family']='survival' if k=='survived' else 'targeted_resource' if target else 'descriptive'
                effects[scene][a][k]=e
    assert len(pairs)==1536
    csvwrite('paired.csv',pairs)
    r=dict(games=len(data),decisions=sum(v['decisions'] for v in data.values()),pairs=len(pairs),design_sha=pin,
        groups=groups,effects=effects,r2_total_contribution=0)
    (H/'comparison.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
    print('verified',len(data),'games',r['decisions'],'actions')
    for scene in scenes:
        print(scene,[(a,groups[scene][a]['alive'],round(groups[scene][a]['drink_wait'],2),round(groups[scene][a]['eat_wait'],2),round(groups[scene][a]['cells'],2)) for a in arms])
if __name__=='__main__':main()
