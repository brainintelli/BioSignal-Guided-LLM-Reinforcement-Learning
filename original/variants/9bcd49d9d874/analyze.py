"""Verify equal budgets, then compare crossed pairs of training seeds and game worlds."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from livepair.figures import _plot_style
from livepair.predictability import outcome
from mea_state2_r2.features import digest

ROOT=Path(__file__).resolve().parent
KEYS=('new_cells','r1_sum')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root,spec,ctx,seed,arm):
    reg=read(root/'registration.json')
    update=read(root/'update.json')
    result=read(root/'result.json')
    manifest=read(root/'checkpoint/manifest.json')
    local_audit=read(root/'replay-audit.json')
    if (reg['study_digest']!=digest(spec) or reg['source_sha']!=ctx['source_sha256'] or
        reg['training_seed']!=seed or reg['arm']!=arm or reg['window']!=[ctx['frame_start'],ctx['frame_end']] or
        update['actor_steps']!=[8] or update['critic_steps']!=[64] or update['initial']!=reg['initial'] or
        any((update['after'][k]!=reg['initial'][k])!=(k in ('actor','critic')) for k in reg['initial']) or
        result['checkpoint_digest']!=digest(manifest) or manifest['registration_sha256']!=digest(reg) or
        manifest['actor_sha256']!=update['after']['actor'] or manifest['update']!=spec['updates']):
        raise ValueError(f'Study binding, checkpoint or budget differs at {root}')
    for name,expected in manifest['files'].items():
        if sha(root/'checkpoint'/name)!=expected:
            raise ValueError(f'Checksum differs at {root}/{name}')
    feedback=dict(positive=0,negative=0,silent=0,sum_contribution=0.,total_decisions=0)
    episodes=spec['episodes'][str(seed)]
    if [r['world'] for r in update['rounds']]!=episodes:
        raise ValueError('Training schedule changed')
    first=[]
    for batch,summary in enumerate(update['rounds'],1):
        rows=[json.loads(line) for line in (root/f'train-{batch}.jsonl').read_text(encoding='utf8').splitlines()]
        if len(rows)!=spec['batch'] or summary['update']['optimizer_steps']!=2:
            raise ValueError('Update budget mismatch')
        for i,r in enumerate(rows):
            trace=outcome(r['transition'])
            expected_sign=(1 if trace['correct'] else -1) if (i+1)%3==0 and not r['terminal'] else 0
            term=sum(.99985**j for j in range(r['duration']))*spec['beta']*expected_sign if arm=='ordered' else 0.
            snap=r['snapshot']
            if (i!=r['index'] or r['seed']!=summary['world'] or expected_sign!=r['r2'] or
                not math.isclose(r['contribution'],term,abs_tol=1e-12) or
                not math.isclose(r['reward'],r['r1_score']['reward']+term,abs_tol=1e-12) or
                snap['events_sha256']!=ctx['events_sha'] or snap['window_index']!=0 or
                snap['features']!=ctx['features'] or snap['source_pin']!=ctx['source_sha256'] or
                r['prepared']['wetware_features']!=ctx['normalized'] or
                (i==0 and any(r['observation']['player'][k]!=v for k,v in spec['starting_reserves'].items()))):
                raise ValueError(f'Training step or fixed MEA context differs at {seed}/{arm}/{batch}/{i}')
            feedback[{1:'positive',-1:'negative',0:'silent'}[expected_sign]]+=1
            feedback['sum_contribution']+=term
            feedback['total_decisions']+=1
            if batch==1:
                first.append([r['snapshot']['context_id'],r['action'],r['r1_score']['reward'],expected_sign])
    ev=result['evaluation']
    if ([v['seed'] for v in ev]!=spec['holdout_seeds'] or local_audit!=dict(seed=seed,arm=arm,worlds=len(ev),decisions=sum(v['decisions'] for v in ev))):
        raise ValueError('Independent Godot audit mismatch')
    return reg,ev,feedback,digest(first)


def signs(values):
    values=np.asarray(values)
    return dict(higher=int((values>1e-10).sum()),tied=int((abs(values)<=1e-10).sum()),lower=int((values< -1e-10).sum()))


def main():
    spec=read(ROOT/'design.json')
    ctx=read(ROOT/'context.json')
    assert digest(ctx)==spec['context_sha']
    seeds=spec['training_seeds']
    worlds=spec['holdout_seeds']
    curves={arm:[] for arm in spec['arms']}
    feedback={}
    table=[]
    config=None
    for j,seed in enumerate(seeds):
        paired_start=None
        for arm in spec['arms']:
            reg,ev,counts,first=verify(ROOT/'models'/str(seed)/arm,spec,ctx,seed,arm)
            if config is None:
                config=(reg['initial'],reg['config'])
            if config!=(reg['initial'],reg['config']) or (paired_start and first!=paired_start):
                raise ValueError('Mismatched initial weights, training hyperparameters or first episode')
            paired_start=first
            feedback[f'{seed}/{arm}']=counts
            curves[arm].append([[w[k] for k in KEYS] for w in ev])
        for i,w in enumerate(worlds):
            record=dict(training_seed=seed,holdout_seed=w)
            for k,name in enumerate(KEYS):
                base=curves['r1'][j][i][k]
                treat=curves['ordered'][j][i][k]
                record.update({name+'_r1':base,name+'_r1_r2':treat,name+'_diff':treat-base})
            table.append(record)
    with (ROOT/'paired-worlds.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    a=np.asarray(curves['r1'])
    b=np.asarray(curves['ordered'])
    delta=b-a
    interval=spec['interval']
    rng=np.random.default_rng(interval['seed'])
    samples=[]
    for _ in range(interval['draws']//500):
        reps=rng.integers(len(seeds),size=(500,len(seeds),1))
        envs=rng.integers(len(worlds),size=(500,1,len(worlds)))
        samples.append(delta[reps,envs].mean(axis=(1,2)))
    bounds=np.quantile(np.concatenate(samples),interval['quantiles'],axis=0)
    per_seed=delta.mean(axis=1)
    summaries={}
    for k,name in enumerate(KEYS):
        summaries[name]=dict(r1=a[:,:,k].mean(),r1_r2=b[:,:,k].mean(),difference=delta[:,:,k].mean(),
                             interval=bounds[:,k].tolist(),by_training_seed=per_seed[:,k].tolist(),
                             training_seed_signs=signs(per_seed[:,k]),paired_world_runs=signs(delta[:,:,k]),
                             holdout_world_signs=signs(delta[:,:,k].mean(axis=0)))
    both=int((per_seed>1e-10).all(axis=1).sum())
    out=dict(metrics=summaries,joint_positive_training_seeds=both,stable=bool(both>=6 and np.all(bounds[0]>0)),
             feedback=feedback,training_decisions_per_arm=8*48,eval_games_per_arm=len(seeds)*len(worlds),
             dataset_sha=ctx['source_sha256'],context_sha=spec['context_sha'],study_digest=digest(spec))
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
    with _plot_style():
        fig=Figure(figsize=(9.2,4.6),dpi=150,layout='constrained')
        for k,(ax,title) in enumerate(zip(fig.subplots(1,2),('每局新增探索格数差','每局原生游戏回报 R1 差'))):
            ax.axvline(0,ls='--',lw=1,color='#777')
            ax.scatter(per_seed[:,k],np.arange(8),color='#236b8b',s=34)
            center=delta[:,:,k].mean()
            ax.plot(bounds[:,k],[8.5,8.5],color='#bc6333',lw=2.5)
            ax.scatter([center],[8.5],color='#bc6333',marker='D')
            ax.set_yticks([*range(8),8.5],[*(f'训练 {i+1}' for i in range(8)),'总体'])
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_xlabel('R1 + 模拟 R2 减去仅 R1')
            ax.grid(axis='y',alpha=.16)
        fig.suptitle('芯片 22021 固定输入 · 8 次配对训练，每次评估 48 个新世界')
        FigureCanvasAgg(fig).print_png(str(ROOT/'effects.png'))
    print(json.dumps({k:v for k,v in out.items() if k!='feedback'}))


if __name__=='__main__':
    main()
