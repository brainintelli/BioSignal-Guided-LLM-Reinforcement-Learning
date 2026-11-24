"""Independent final-only arithmetic/evidence check; no model/device imports."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

ARMS=('none','before','after')
REGISTRATION='b520f2e1ef88048c2735fba2fc3a6a6ad3b31abf870c9245c753462d406290a1'
SOURCE='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'

def read(p):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate JSON key')
            result[k]=v
        return result
    def invalid(value):raise ValueError('Nonfinite JSON')
    return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=invalid)

def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def canonical_sha(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def paired(values):
    if len(values)!=8 or any(not math.isfinite(v) for v in values):raise ValueError('Need 8 finite paired differences')
    observed=statistics.mean(values);tol=1e-12*max(1.,abs(observed))
    tail=0
    for mask in range(256):
        permuted=statistics.mean(v if mask&(1<<i) else -v for i,v in enumerate(values))
        tail+=abs(permuted)>=abs(observed)-tol
    rng=random.Random(6800100)
    draws=sorted(statistics.mean(rng.choices(values,k=8)) for _ in range(20000))
    def quantile(q):
        pos=q*(len(draws)-1);lo=int(pos);hi=min(lo+1,len(draws)-1)
        return draws[lo]+(draws[hi]-draws[lo])*(pos-lo)
    return dict(differences=values,mean_difference=observed,bootstrap_95=[quantile(.025),quantile(.975)],
                p_two_sided=tail/256,p_bonferroni_two_contrasts=min(1.,2*tail/256))

def close(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b

def audit(root):
    verified=read(root/'delivery_verified.json')
    if verified.get('passed') is not True:raise ValueError('Final delivery not verified')
    d=root/'delivery';manifest_path=d/'accepted_campaign_manifest.json';manifest=read(manifest_path)
    if sha(manifest_path)!=verified['accepted_manifest_sha256']:raise ValueError('Manifest pin changed')
    for k,v in dict(passed=True,cells=24,training_decisions=9216,registration_sha256=REGISTRATION,source_pin=SOURCE).items():
        if manifest.get(k)!=v:raise ValueError('Campaign acceptance '+k)
    checked=0
    for rel,item in manifest['files'].items():
        p=(d/rel).resolve()
        if not p.is_relative_to(d.resolve()) or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Artifact changed: '+rel)
        checked+=1
    exp=d/'experiment01';status=read(exp/'batch_status.json');saved=read(exp/'analysis.json');reg=read(exp/'registration.json')
    if status.get('passed') is not True or status.get('complete') is not True or saved.get('status')!='complete':raise ValueError('Not complete')
    if canonical_sha(reg)!=REGISTRATION or reg['source_pin']!=SOURCE:raise ValueError('Registration changed')
    if len(status['cells'])!=24 or any(len(c['stages'])!=3 or any(s['exit_code']!=0 for s in c['stages']) for c in status['cells']):
        raise ValueError('Missing successful stage')
    if {(c['block'],c['arm']) for c in status['cells']}!=set(itertools.product(range(8),ARMS)):raise ValueError('Cell inventory')
    cells={};descriptive={a:[] for a in ARMS};train_decisions=0
    for block,arm in itertools.product(range(8),ARMS):
        folder=exp/f'b{block:02d}_{arm}';ev=read(folder/'evaluation.json');tr=read(folder/'train/report.json')
        receipt=read(exp/f'verified_b{block:02d}_{arm}.json')
        for k,v in dict(passed=True,block=block,arm=arm,registration_sha256=REGISTRATION,source_pin=SOURCE,fresh_process=True,training_decisions=384).items():
            if receipt.get(k)!=v:raise ValueError('Independent receipt '+k)
        if tr['initial']!=reg['common_initial'] or ev['checkpoint_tensors']['pre']!=reg['common_initial'] or ev['checkpoint_tensors']['post']!=tr['final']:
            raise ValueError('Common/checkpoint tensors')
        if tr['decisions']!=384 or len(tr['rounds'])!=3 or any(r['applied_steps']!={'actor':2,'critic':16} for r in tr['rounds']):
            raise ValueError('Unequal training budget')
        if ev['r2_enabled'] is not False or ev['realtime_training'] is not False or ev['device_commands_sent']!=0 or ev['alpha']!=1:
            raise ValueError('Scope/R1-only violation')
        train_decisions+=tr['decisions']
        cells[block,arm]={p:statistics.mean(ev['metrics'][p][c]['discounted_r1'] for c in ('before','after')) for p in ('pre','post')}
        descriptive[arm].extend(ev['metrics']['post'].values())
    contrasts={}
    for control in ('before','none'):
        diffs=[cells[b,'after']['post']-cells[b,control]['post'] for b in range(8)]
        gains=[diffs[b]-(cells[b,'after']['pre']-cells[b,control]['pre']) for b in range(8)]
        value=dict(post=paired(diffs),gain=paired(gains))
        if not close(value,saved['contrasts']['after_minus_'+control]):raise ValueError('Independent statistics differ')
        contrasts['after_minus_'+control]=value
    positive=all(c['post']['mean_difference']>0 and c['post']['bootstrap_95'][0]>0 and c['post']['p_bonferroni_two_contrasts']<.05 for c in contrasts.values())
    if positive!=saved['positive_evidence']:raise ValueError('Decision rule differs')
    classification=('registered_exploratory_advantage' if positive else
                    'direction_only_not_positive_evidence' if all(c['post']['mean_difference']>0 for c in contrasts.values()) else
                    'no_consistent_advantage_across_both_controls')
    means={a:{p:statistics.mean(cells[b,a][p] for b in range(8)) for p in ('pre','post')} for a in ARMS}
    if not close(means,saved['arms']):raise ValueError('Group means differ')
    return dict(passed=True,verified_files=checked,cells=24,training_decisions=train_decisions,
      registration_sha256=REGISTRATION,source_pin=SOURCE,arms=means,contrasts=contrasts,
      positive_evidence=positive,classification=classification,biological_replications=1,biological_causality_verified=False,
      secondary={a:{k:statistics.mean(x[k] for x in values) for k in ('final_health','final_hunger','final_thirst','visited_count','elapsed_ticks')} for a,values in descriptive.items()},
      natural_termination_counts={a:sum(x['terminated'] for x in values) for a,values in descriptive.items()})

def selftest():
    p=paired([1.]*8);assert p['p_two_sided']==2/256 and p['p_bonferroni_two_contrasts']==4/256
    assert p['bootstrap_95']==[1.,1.]
    assert paired([0.]*8)['p_two_sided']==1
    assert paired([-1.,1.]*4)['p_two_sided']==1
    assert paired([-1.]*8)['p_two_sided']==p['p_two_sided']
    try:paired([1.]*7)
    except ValueError:pass
    else:raise AssertionError('incomplete pairs accepted')
    print('Arithmetic boundary checks passed; no experimental outcomes loaded')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:selftest()
    else:
        result=audit(a.root)
        with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps({k:result[k] for k in ('passed','cells','training_decisions','positive_evidence','classification')},ensure_ascii=False))
