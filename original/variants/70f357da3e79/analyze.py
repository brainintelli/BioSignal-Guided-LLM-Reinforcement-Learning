"""Verify the two matched PPO runs and aggregate paired differences by holdout world."""
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from livepair.figures import _plot_style
from livepair.predictability import outcome
from mea_state2_r2.features import digest, transform

ROOT = Path(__file__).resolve().parent
METRICS = ('new_cells', 'r1_sum')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def counts(delta):
    delta=np.asarray(delta)
    return dict(higher=int((delta>1e-9).sum()),same=int((np.abs(delta)<=1e-9).sum()),lower=int((delta< -1e-9).sum()))


def verify_run(study, data, seed, arm):
    root=ROOT/'models'/str(seed)/arm
    reg=read(root/'registration.json')
    result=read(root/'result.json')
    manifest=read(root/'checkpoint/manifest.json')
    checks = [
        reg['training_seed']==seed, reg['arm']==arm, reg['study_digest']==digest(study),
        reg['eval_seeds']==study['holdout_seeds'], reg['train_order']==study['train_trials'],
        reg['test_order']==study['eval_trials'], reg['source_sha']==digest(data)==study['replay_digest'],
        reg['normalization_id']==digest(data['normalization']), result['optimizer']['optimizer_steps']==2,
        result['train_decisions']==12, result['holdout_windows']==6, result['arm']==arm,
        manifest['actor_sha256']==result['after']['actor'],digest(manifest)==result['checkpoint_sha'],
        manifest['registration_sha256']==digest(reg), reg['initial']==result['initial'],
        reg['real_game_contingent_outputs']==0, reg['starting_scene']==study['starting_scene'],
    ]
    if not all(checks):
        raise ValueError(f'Unexpected source, split, budget or model in {root}')
    for k in ('actor','critic','base','reference'):
        if (reg['initial'][k]!=result['after'][k])!=(k in ('actor','critic')):
            raise ValueError(f'Unexpected change in {k}')
    for name,sha in manifest['files'].items():
        with (root/'checkpoint'/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=sha:
                raise ValueError(f'Checkpoint checksum differs for {name}')
    rows=[json.loads(s) for s in (root/'train.jsonl').read_text(encoding='utf8').splitlines()]
    start=rows[0]['observation']['player']
    if any(start[k]!=v for k,v in study['starting_reserves'].items()) or len(rows)!=12:
        raise ValueError('Incorrect starting reserves or training duration')
    training=[]
    for index,(row,w) in enumerate(zip(rows,data['trials'])):
        sign=row['reward_record']['simulated_r2']
        trace=outcome(row['transition'])
        eligible=(index+1)%3==0 and not row['terminal']
        expected=(1 if trace['correct'] else -1) if eligible else 0
        weighted=sum(.99985**i for i in range(row['duration']))*sign*.001 if arm=='ordered' else 0.
        snap=row['snapshot']
        if (sign!=expected or row['reward_record']['actual_feedback_to_game'] or
            not math.isclose(row['reward']-row['r1_score']['reward'],weighted,abs_tol=1e-12) or
            snap['window_index']!=w['index'] or snap['events_sha256']!=w['events_sha'] or
            row['prepared']['wetware_features']!=list(transform(w['features'],data['normalization']))):
            raise ValueError(f'Replay or R2 mismatch at step {index+1}')
        training.append(dict(step=index+1,context=snap['context_id'],action=row['action'],sign=sign,weight=weighted,
                             **trace))
    if [r['seed'] for r in result['eval']]!=study['holdout_seeds']:
        raise ValueError('Holdout order mismatch')
    for episode in result['eval']:
        rollout=read(root/f"eval-{episode['seed']}.json")
        expected=study['eval_trials'][:len(rollout)]
        if ([r['index'] for r in rollout]!=expected or len(rollout)!=episode['decisions'] or
            not math.isclose(sum(r['r1'] for r in rollout),episode['r1_sum']) or
            episode['visited_count']!=rollout[-1]['after']['visited_count']):
            raise ValueError('Invalid evaluation rollout')
    return reg,training,result['eval']


def main():
    study=read(ROOT/'design.json')
    data=read(ROOT.parent/'ppo-data.json')
    if set(study['holdout_seeds']) & (set(study['training_seeds']+study['pilot_seeds'])|set(range(840001,840009))):
        raise ValueError('A holdout seed was reused')
    results={}
    feedback={}
    ref=None
    per_seed={key:[] for key in METRICS}
    for seed in study['training_seeds']:
        e={}
        train={}
        for arm in study['arms']:
            reg,train[arm],e[arm]=verify_run(study,data,seed,arm)
            identity=(reg['initial'],reg['config'],reg['source_sha'])
            if ref is not None and ref!=identity:
                raise ValueError('Initialization or budget differs')
            ref=identity
        if [{k:v for k,v in r.items() if k!='weight'} for r in train['r1']]!=[{k:v for k,v in r.items() if k!='weight'} for r in train['ordered']]:
            raise ValueError('The pair did not train on the same trajectory')
        eligible=[r for r in train['ordered'] if r['sign']]
        feedback[str(seed)]=dict(opportunities=eligible,contribution=sum(r['weight'] for r in eligible))
        results[str(seed)]={}
        for key in METRICS:
            x=np.array([v[key] for v in e['r1']])
            y=np.array([v[key] for v in e['ordered']])
            delta=y-x
            per_seed[key].append(delta)
            results[str(seed)][key]=dict(r1=float(x.mean()),r1_r2=float(y.mean()),delta=float(delta.mean()),
                                        **counts(delta),by_world=delta.tolist())
    rng=np.random.default_rng(study['interval']['seed'])
    n=len(study['holdout_seeds'])
    draws=rng.integers(0,n,size=(study['interval']['draws'],n))
    summary={}
    for key in METRICS:
        delta=np.mean(per_seed[key],axis=0)
        lower,upper=np.percentile(delta[draws].mean(axis=1),[1.25,98.75])
        summary[key]=dict(r1=mean(r[key]['r1'] for r in results.values()),
                          r1_r2=mean(r[key]['r1_r2'] for r in results.values()),delta=float(delta.mean()),
                          interval=[float(lower),float(upper)],**counts(delta),
                          passes=bool(lower>0 and all(r[key]['delta']>0 for r in results.values())))
    report=dict(feedback=feedback,summary=summary,training_pairs=results,world_seeds=study['holdout_seeds'],
                stable_both=all(x['passes'] for x in summary.values()),n_training_pairs=2,n_eval_worlds=n,n_organoids=1)
    (ROOT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    with _plot_style():
        figure=Figure(figsize=(9,3.9),dpi=150,layout='constrained')
        a,b=figure.subplots(1,2)
        for ax,key,label in [(a,'new_cells','新增探索格数'),(b,'r1_sum','游戏原生回报 R1')]:
            for i,r in enumerate(results.values()):
                off=-.18 if i==0 else .18
                heights=[r[key][name] for name in ('lower','same','higher')]
                bars=ax.bar(np.arange(3)+off,heights,width=.34,color=['#236983','#bb673d'][i],label=f'训练重复 {i+1}')
                ax.bar_label(bars,padding=3,fontsize=9)
            ax.set_title(label)
            ax.set_xticks(np.arange(3),['加入 R2 后较低','持平','较高'])
            ax.set_ylim(0,n+6)
            ax.set_ylabel('留存游戏种子数')
        b.legend(loc='upper right',fontsize=8)
        figure.suptitle(f'同一初始场景和训练预算 · {n} 个新的配对游戏种子')
        FigureCanvasAgg(figure).print_png(str(ROOT/'holdout.png'))
    print(json.dumps(dict(summary=summary,per_pair={s:{k:{a:b for a,b in v.items() if a!="by_world"} for k,v in d.items()} for s,d in results.items()})))


if __name__=='__main__':
    main()
