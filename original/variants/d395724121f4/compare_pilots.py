"""Compare paired evidence without drawing conclusions from four worlds."""
import argparse
from collections import Counter
import gzip
from itertools import zip_longest
import json
from pathlib import Path

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)

def compare_pair(a,b,fields):
    count=0;changed_rewards=0;delta=0.
    for x,y in zip_longest(rows(a),rows(b)):
        if x is None or y is None:raise ValueError('Paired trace lengths differ')
        for field in fields:
            if x[field]!=y[field]:raise ValueError(f'Paired initial behavior differs at record{count}:{field}')
        d=y['reward']['reward']-x['reward']['reward']
        changed_rewards+=abs(d)>1e-12;delta+=d;count+=1
    return dict(records=count,identical_behavior=True,reward_changed_records=changed_rewards,mean_reward_delta=delta/count)

def risk_summary(path):
    counts=Counter();episodes={}
    for r in rows(path):
        obs=r['observation'];p=obs['player'];action=r['action'];counts['decisions']+=1
        hungry=p['hunger']<40 and obs['inventory']['berries']==0
        counts['hungry_without_food']+=hungry
        counts['low_thirst']+=p['thirst']<40
        counts['near_edge3']+=min(p['x'],p['y'],95-p['x'],95-p['y'])<=3
        if hungry:
            counts['hungry_'+action['verb']]+=1
            foods={x['id'] for x in obs['nearby'] if x['kind'] in ('berry','mushroom') and x.get('available',True)}
            counts['hungry_known_food']+=bool(foods)
            counts['hungry_food_navigation']+=action['verb']=='navigate' and action.get('target_id') in foods
    return dict(counts)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    base=Path(a.root);control=base/'supply_pilot_control_v2';treatment=base/'supply_pilot_treatment_v2'
    fields=('env','seed','observation','action','transition','next_observation','logp','verb_index','target_index')
    result=dict(initial_evaluation=compare_pair(control/'initial_evaluation.jsonl.gz',treatment/'initial_evaluation.jsonl.gz',fields),
        first_rollout=compare_pair(control/'rollout_0001.jsonl.gz',treatment/'rollout_0001.jsonl.gz',fields),
        interpretation='Paired engineering and short-run risk diagnostic; four worlds do not establish superior reward or milestone completion',arms={})
    for label,root in [('control',control),('treatment',treatment)]:
        report=json.loads((root/'selection_0002.json').read_text())
        result['arms'][label]=dict(summary=report['summary'],risk=risk_summary(root/'selection_0002.jsonl.gz'))
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
