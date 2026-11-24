"""Freeze the R1 reference on historical pure-policy training data before running any new arm."""
import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mea_state2_r2'))
from mea_state2_r2.reward_mix import discounted_ticks,validate_mix


def read(p):return json.loads(p.read_bytes())


def save(p,value):
    payload=json.dumps(value,indent=2,allow_nan=False).encode()
    if p.exists():
        if p.read_bytes()!=payload:raise ValueError('Registration changed: '+str(p))
    else:p.write_bytes(payload)


def register(folder):
    root=folder.parent
    prior=root/'factorial-tick48'
    tri=root/'three-arm-six-seeds'
    protocol=read(prior/'protocol.json')
    rates=[]
    hashes={}
    old={k:[] for k in ('ordered','shuffled','r2only')}
    for cohort,seeds in protocol['cohorts'].items():
        for s in seeds:
            path=tri/cohort/'pure'/str(s)/'training-trace.json'
            hashes[str(s)]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows=read(path)
            if len(rows)!=96 or any(r['r2_contribution'] or r['reward']!=r['r1'] for r in rows):
                raise ValueError('Pure training reference contaminated')
            rates.extend(abs(r['r1'])/discounted_ticks(r['duration'],.99985) for r in rows if not r['terminal'] and r['r1']!=0)
            for arm in old:
                orig=(prior/cohort if arm=='r2only' else root if cohort=='original' else tri/cohort)
                old[arm].extend(read(orig/arm/str(s)/'training-trace.json'))
    mix=validate_mix(dict(version='r1-relative-v1',r1_weight=1.,r2_weight=.1,amplitude_fraction=.5,
                          reference_per_tick=statistics.median(rates)))
    legacy={}
    for arm,rows in old.items():
        x=[abs(r['r1']) for r in rows];y=[abs(r['r2_contribution']) for r in rows]
        legacy[arm]=dict(n=len(x),label_values=sorted(set(r['r2'] for r in rows if r['r2'] is not None)),
            median_abs_r1=statistics.median(x),min_abs_contribution=min(y),max_abs_contribution=max(y),
            dominant_steps=sum(v>u for u,v in zip(x,y)),sum_abs_ratio=sum(y)/sum(x),
            max_step_ratio=max(v/u for u,v in zip(x,y) if u>0),total_reward_sign_flips=sum(r['reward']*r['r1']<0 for r in rows))
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'calibration.json',dict(mix=mix,training_trace_hashes=hashes,valid_reference_steps=len(rates),
                                      reference='median nonzero nonterminal abs(pure R1)/discounted elapsed ticks',
                                      legacy_training_audit=legacy))
    protocol.update(reward_mix=mix,rerun_arms=['ordered','shuffled','r2only'],
        assignments={'original':{'ordered':'16','r2only':'23','shuffled':'23'},
                     'added':{'ordered':'23','r2only':'16','shuffled':'16'}},
        unchanged_controls=['pure','stateonly'])
    save(folder/'protocol.json',protocol)
    for cohort in protocol['cohorts']:
        src=prior/cohort
        d=folder/cohort
        d.mkdir(exist_ok=True)
        for name in ('replay.json','preceding-quiet.json'):
            if not (d/name).exists():shutil.copyfile(src/name,d/name)
        b=read(src/'budget.json')
        b['reward_mix']=mix
        save(d/'budget.json',b)
    print(json.dumps(mix))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    register(p.parse_args().folder)
