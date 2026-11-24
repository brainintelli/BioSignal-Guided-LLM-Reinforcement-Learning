"""Audit budgets, actor and critic inputs, and estimate paired effects on two held-out endpoints."""
import csv
import hashlib
import json
from pathlib import Path
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from livepair.figures import _plot_style
from mea_state2_r2.features import digest, NAMES

ROOT=Path(__file__).resolve().parent
LABELS={'masked':'屏蔽 MEA','sham':'零输出 MEA','regular':'规律刺激后 MEA','irregular':'随机刺激后 MEA'}
NAMES_ZH=('每局新增探索格数','每局游戏回报 R1')

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]

def check_model(spec,ctx,seed,arm,root=ROOT):
    p=root/'models'/str(seed)/arm
    reg=read(p/'registration.json'); up=read(p/'update.json'); res=read(p/'result.json')
    cp=read(p/'checkpoint/manifest.json'); audit=read(p/'replay-audit.json'); meta=read(p/'checkpoint/policy.json')
    series=ctx['series']
    if (reg['study_digest']!=digest(spec) or reg['windows_digest']!=digest(series) or reg['source_sha']!=spec['source_sha'] or
        reg['initial']!=up['initial'] or (reg['training_seed'],reg['arm'])!=(seed,arm) or reg['config']['alpha']!=1 or
        up['actor_steps']!=[8] or up['critic_steps']!=[64] or cp['registration_sha256']!=digest(reg) or
        cp['update']!=4 or cp['actor_sha256']!=up['after']['actor'] or digest(cp)!=res['checkpoint_digest'] or
        any((up['after'][k]!=v)!=(k in ('actor','critic')) for k,v in reg['initial'].items())):
        raise ValueError(f'Training budget, initial parameters or checkpoint changed at {p}')
    if meta['comparison']!=arm or meta['r2_enabled'] or meta['beta']!=0 or meta['state2_source']!=ctx['state2_source']:
        raise ValueError('Checkpoint input or reward contract differs')
    for k,v in cp['files'].items():
        with (p/'checkpoint'/k).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=v:raise ValueError(f'Checkpoint digest mismatch {k}')
    for n,(world,batch) in enumerate(zip(spec['episodes'][str(seed)],up['rounds']),1):
        steps=lines(p/f'train-{n}.jsonl')
        assert len(steps)==batch['decisions']==spec['batch'] and world==batch['world'] and batch['update']['optimizer_steps']==2
        for i,r in enumerate(steps):
            w=series['train'][(n-1)*spec['batch']+i]
            consumed=[0.]*4 if arm=='masked' else w['normalized']
            snap=r['snapshot']; prep=r['prepared']
            text=','.join(f'{key}={value:.3f}' for key,value in zip(NAMES,consumed))
            assert snap['source_pin']==spec['source_sha'] and snap['events_sha256']==w['events_sha'] and snap['window_index']==w['index']
            assert snap['features']==w['features'] and snap['normalized']==w['normalized'] and prep['wetware_features']==consumed
            assert [snap['window_start'],snap['window_end']]==[w['frame_start'],w['frame_end']] and prep['critic_input'][-4:]==consumed
            assert text in prep['messages'][1]['content'] and r['seed']==world and r['index']==i and r['contribution']==0 and r['reward']==r['r1_score']['reward']
            if i==0:assert all(r['observation']['player'][k]==v for k,v in spec['starting_reserves'].items())
    assert len(up['rounds'])==spec['updates']
    held=lines(p/'holdout.jsonl'); ev=res['evaluation']; worlds=spec['holdout_seeds']
    assert [g['world'] for g in held]==[r['seed'] for r in ev]==worlds
    for g,r in zip(held,ev):
        assert 0<len(g['rows'])==r['decisions']<=spec['eval_decisions']
        for row,w in zip(g['rows'],series['holdout']):
            assert row['wetware_window']==w['index'] and row['wetware_events_sha']==w['events_sha']
            assert row['consumed_features']==([0.]*4 if arm=='masked' else w['normalized'])
    assert audit==dict(seed=seed,arm=arm,worlds=len(ev),decisions=sum(r['decisions'] for r in ev))
    return [reg['initial'],reg['config']], [[r[k] for k in spec['endpoints']] for r in ev],audit,[digest([row['action'] for row in game['rows']]) for game in held]

def summarize(x,bounds):
    return dict(difference=float(x.mean()),interval=bounds.tolist(),by_seed=x.mean(axis=1).tolist(),
                improved_pairs=int((x>1e-10).sum()),tied_pairs=int((abs(x)<=1e-10).sum()),lower_pairs=int((x< -1e-10).sum()))

def plot(res,contrasts,metrics):
    names=['当轮零输出 MEA\n相对屏蔽 MEA','规律刺激后\n相对零输出','多点随机刺激后\n相对零输出','首次刺激前自发 MEA\n相对屏蔽 MEA（复核）']
    with _plot_style():
        fig=Figure(figsize=(10.4,5.2),dpi=145,layout='constrained')
        axs=fig.subplots(1,2)
        for k,ax in enumerate(axs):
            ax.axvline(0,color='#56616a',ls='--',lw=1)
            for j,(a,b) in enumerate(contrasts):
                x=res['contrasts'][a+'-'+b]['metrics'][metrics[k]]
                ax.scatter(x['by_seed'],j+.09+np.linspace(0,.15,8),s=17,alpha=.65,color='#286884')
                ax.plot(x['interval'],[j,j],color='#b75a37',lw=2.5)
                ax.scatter([x['difference']],[j],s=40,marker='D',color='#b75a37',zorder=3)
            ax.set_yticks(range(len(names)),names if k==0 else ['']*len(names))
            ax.set_ylim(3.6,-.38);ax.grid(axis='y',alpha=.15)
            ax.set_title(NAMES_ZH[k]);ax.set_xlabel('相对对照的配对差')
        fig.suptitle('芯片 22021 · 8 个训练种子 × 48 个新留存游戏世界')
        fig.legend(handles=[Line2D([],[],marker='o',lw=0,color='#286884',label='单个训练种子'),Line2D([],[],marker='D',lw=2,color='#b75a37',label='总体均差及校正区间')],loc='outside lower center',ncol=2,fontsize=8)
        FigureCanvasAgg(fig).print_png(ROOT/'effects.png')

def main():
    spec=read(ROOT/'design.json'); seeds=spec['training_seeds']; worlds=spec['holdout_seeds']; arms=spec['arms']
    base=read(ROOT/'pre-stim-baseline/design.json')
    assert base['parent_sha']==digest(spec) and base['holdout_seeds']==worlds and base['episodes']==spec['episodes']
    metrics=spec['endpoints']; contrasts=spec['contrasts']+base['contrasts']; arms=arms+base['arms']
    ref=None; results={}; audits=[]; sequences={}
    for a in arms:
        folder=ROOT/'pre-stim-baseline' if a=='baseline' else ROOT
        contract=base if a=='baseline' else spec
        ctx=read(folder/(a+'.json'))
        assert digest(ctx)==contract['contexts'][a] and ctx['source_sha256']==spec['source_sha']
        vals=[]; sequences[a]=[]
        for s in seeds:
            initial,ev,receipt,actions=check_model(contract,ctx,s,a,folder)
            sequences[a].append(actions)
            if ref is None:ref=initial
            assert ref==initial
            vals.append(ev); audits.append(receipt)
        results[a]=np.asarray(vals)
    check=read(ROOT/'masked.json'); sham=read(ROOT/'sham.json');assert check['series']==sham['series']
    with (ROOT/'paired-worlds.csv').open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f);out.writerow(['train_seed','holdout_world']+[f'{a}_{k}' for a in arms for k in metrics]+[f'{a}_minus_{b}_{k}' for a,b in contrasts for k in metrics])
        for j,s in enumerate(seeds):
            for i,w in enumerate(worlds):
                out.writerow([s,w]+[v for a in arms for v in results[a][j,i]]+[v for a,b in contrasts for v in (results[a]-results[b])[j,i]])
    delta=np.stack([results[a]-results[b] for a,b in contrasts],axis=2)
    rng=np.random.default_rng(spec['interval']['seed']); boot=[]
    for _ in range(spec['interval']['draws']//200):
        si=rng.integers(len(seeds),size=(200,len(seeds),1));wi=rng.integers(len(worlds),size=(200,1,len(worlds)))
        boot.append(delta[si,wi].mean(axis=(1,2)))
    ci=np.quantile(np.concatenate(boot),spec['interval']['quantiles'],axis=0)
    aux=[]; rng=np.random.default_rng(base['interval']['seed'])
    for _ in range(base['interval']['draws']//200):
        si=rng.integers(len(seeds),size=(200,len(seeds),1)); wi=rng.integers(len(worlds),size=(200,1,len(worlds)))
        aux.append(delta[si,wi,-1].mean(axis=(1,2)))
    ci[:,-1]=np.quantile(np.concatenate(aux),base['interval']['quantiles'],axis=0)
    out=dict(source_sha=spec['source_sha'],study_sha=digest(spec),arm_means={a:dict(zip(metrics,v.mean(axis=(0,1)).tolist())) for a,v in results.items()},
             contrasts={},audited_models=len(audits),audited_games=sum(x['worlds'] for x in audits),audited_actions=sum(x['decisions'] for x in audits),
             training_decisions_per_arm=len(seeds)*spec['updates']*spec['batch'],primary_endpoints=len(spec['contrasts'])*len(metrics),sensitivity_endpoints=2,sensitivity_sha=digest(base))
    for d,(a,b) in enumerate(contrasts):
        v=delta[:,:,d]; jointly=int(np.all(v.mean(axis=1)>1e-10,axis=1).sum())
        out['contrasts'][a+'-'+b]=dict(jointly_positive=jointly,confidence=(.975 if a=='baseline' else 1-.05/6),identical_action_sequences=int((np.array(sequences[a])==np.array(sequences[b])).sum()),stable=bool(jointly>=6 and np.all(ci[0,d]>0)),
            metrics={k:summarize(v[:,:,i],ci[:,d,i]) for i,k in enumerate(metrics)})
    (ROOT/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    plot(out,contrasts,metrics)
    print(json.dumps({k:v for k,v in out.items() if k!='contrasts'}))
    for key,v in out['contrasts'].items():print(key,v['jointly_positive'],v['stable'],[(x['difference'],x['interval']) for x in v['metrics'].values()])
if __name__=='__main__':main()
