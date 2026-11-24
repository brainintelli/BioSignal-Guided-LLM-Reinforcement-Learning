"""Plot the measured single-pulse classifier and matched held-out game results."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.tools'/'mea_baseline_deps')]+[str(ROOT/'standalone'/x) for x in
    ('mea_online','mea_live_pair','mea_state2_r2','mea_binary_r2','mea_stimulation_v2')]
import numpy as np
from matplotlib.figure import Figure
from livepair.figures import _plot_style,_save

def read(path):return json.loads(path.read_text(encoding='utf8'))

def signals(folder):
    data=read(folder/'replay.json')
    with _plot_style():
        fig=Figure(figsize=(8.8,4.6),layout='constrained',dpi=155)
        fig.suptitle('芯片 22554 · 探索性单脉冲的三位点识别\n每次留出一整组，用另外两组预测',fontsize=14)
        for ax,k,title in zip(fig.subplots(1,2),('pre','post'),('刺激前背景（−3 至 −2 秒）','刺激后回读（12 至 13 秒）')):
            c=data['exploratory_classification'][k]
            matrix=np.array(c['confusion'])
            ax.imshow(matrix,vmin=0,vmax=3,cmap='Blues',interpolation='nearest')
            for (i,j),v in np.ndenumerate(matrix):
                ax.text(j,i,str(v),ha='center',va='center',fontsize=16,color='white' if v>=2 else '#21374b')
            ax.set(xticks=range(4),xticklabels=['H','W','F','未判定'],yticks=range(3),yticklabels=list('HWF'),
                   xlabel='预测位点',ylabel='实际位点',title=f"{title}\n正确 {c['correct']}/9，置换 p = {c['p_exact']:.3f}")
            ax.spines[:].set_visible(False)
        _save(fig,folder/'single-pulse-confusion.png')

def games(folder):
    results={arm:read(folder/arm/'result.json') for arm in ('ordered','shuffled')}
    a,b=results.values()
    if (a['study'],a['replay_digest'])!=(b['study'],b['replay_digest']):raise ValueError('Different recording or registration')
    metrics={'explored':'探索新格数','health':'生命值','hunger':'饱食','thirst':'水分','r1':'游戏回报 R1','death':'死亡率'}
    paired=[];seeds=[r['training_seed'] for r in a['seeds']]
    if seeds!=[r['training_seed'] for r in b['seeds']]:raise ValueError('Training seeds differ')
    for x,y in zip(a['seeds'],b['seeds']):
        if x['initial']!=y['initial'] or x['update']['world_size']!=8 or y['update']['world_size']!=8:
            raise ValueError('Paired model initialization or eight-rank update differs')
        if any(not d['update']['actor_changed'] or not d['update']['critic_changed'] for d in (x,y)):
            raise ValueError('Optimizer did not update model')
        if [e['seed'] for e in x['eval']]!=[e['seed'] for e in y['eval']]:raise ValueError('Held-out seeds differ')
        for ex,ey in zip(x['eval'],y['eval']):
            paired.append(dict(training_seed=x['training_seed'],game_seed=ex['seed'],ordered=ex,shuffled=ey,
                               delta={k:ex[k]-ey[k] for k in metrics}))
    summary={}
    for k in metrics:
        byseed=[float(np.mean([r['delta'][k] for r in paired if r['training_seed']==seed])) for seed in seeds]
        deltas=[r['delta'][k] for r in paired]
        summary[k]=dict(ordered=float(np.mean([r['ordered'][k] for r in paired])),shuffled=float(np.mean([r['shuffled'][k] for r in paired])),
                        difference=float(np.mean(deltas)),training_seed_differences=byseed,
                        positive=sum(x>0 for x in deltas),equal=sum(x==0 for x in deltas),negative=sum(x<0 for x in deltas))
    (folder/'paired-game-comparison.json').write_text(json.dumps(dict(summary=summary,pairs=paired,training_seeds=seeds,
        independent_chip_count=1,heldout_physical_trials=3),ensure_ascii=False,indent=2),encoding='utf8')
    with _plot_style():
        fig=Figure(figsize=(10.4,5.5),layout='constrained',dpi=155)
        fig.suptitle('芯片 22554 · 有序与打乱信号的离线 PPO 评估\n横轴为有序减打乱，每个点汇总同一训练种子的 8 个留存游戏种子',fontsize=14)
        for ax,(k,name) in zip(fig.subplots(2,3).ravel(),metrics.items()):
            row=summary[k];d=row['training_seed_differences']
            limit=max(max(abs(x) for x in d)*1.35,0.05 if k in ('r1','death') else .25)
            ax.axvline(0,c='#878d99',lw=1,ls='--')
            ax.scatter(d,[2,1,0],s=46,c=['#1d5e85','#258369','#9d552c'],zorder=3)
            ax.set(xlim=(-limit,limit),ylim=(-.6,2.6),yticks=[2,1,0],yticklabels=['种子 1','种子 2','种子 3'],
                   title=f"{name}   平均差 {row['difference']:+.3g}")
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        _save(fig,folder/'paired-game-results.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    signals(args.folder)
    if all((args.folder/a/'result.json').exists() for a in ('ordered','shuffled')):games(args.folder)
