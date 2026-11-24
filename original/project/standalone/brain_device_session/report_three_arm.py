"""Pair pure R1, ordered MEA and phase-preserving shuffle by model and game seed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from report_long_recorded import Figure, NAMES, _plot_style, _save, inspect_arm, np, read, require

GROUPS={'pure':'纯模型 · R1','ordered':'有序 MEA','shuffled':'打乱 MEA'}
COLORS={'pure':'#474f69','ordered':'#08776d','shuffled':'#b15922'}
STYLES={'pure':'-','ordered':'--','shuffled':':'}
METRICS=(*NAMES,'death','r1','decisions','ticks')
PAIRS=(('ordered','pure'),('ordered','shuffled'),('shuffled','pure'))


def summarize(rows,seeds):
    require(len(set((r['training_seed'],r['game_seed']) for r in rows))==len(rows),'Game pairs duplicated')
    means={a:{k:mean(r[a][k] for r in rows) for k in METRICS} for a in GROUPS}
    effects={}
    for a,b in PAIRS:
        contrasts={}
        for k in METRICS:
            values=[r[a][k]-r[b][k] for r in rows]
            by_seed=[mean(r[a][k]-r[b][k] for r in rows if r['training_seed']==s) for s in seeds]
            contrasts[k]=dict(mean=mean(values),median=median(values),by_seed=by_seed,
                higher_equal_lower=[sum(x>1e-9 for x in values),sum(abs(x)<=1e-9 for x in values),
                                    sum(x< -1e-9 for x in values)],
                seeds_positive=sum(v>1e-9 for v in by_seed))
        effects[a+'-'+b]=contrasts
    per_100_ticks={a:100*sum(r[a]['explored'] for r in rows)/sum(r[a]['ticks'] for r in rows)
                   for a in GROUPS}
    return dict(training_seeds=seeds,paired_games=len(rows),means=means,effects=effects,
                explored_per_100_elapsed_game_ticks=per_100_ticks,
                completion={a:dict(at48=sum(r[a]['decisions']==48 for r in rows),
                                   deaths=sum(r[a]['death'] for r in rows)) for a in GROUPS},
                health_during_games={a:dict(minimum=min(r['health_path'][a]['lowest'] for r in rows),
                    games_below_initial=sum(r['health_path'][a]['lowest']<r['health_path'][a]['initial']
                                            for r in rows)) for a in GROUPS},
                stable_ordered=all(effects['ordered-'+arm]['explored']['seeds_positive']==len(seeds)
                                   for arm in ('pure','shuffled')))


def analyze(folder):
    prior=folder.parent
    protocol=read(folder/'protocol.json')
    cohorts=protocol['cohorts']
    seeds=cohorts['original']+cohorts['added']
    require(len(seeds)==len(set(seeds)) and len(protocol['game_eval_seeds'])==8,'Seed registration differs')
    source=read(prior.parent/'raw-source-audit.json')
    require(source['replay_sha256']==protocol['source_sha'],'Independent source audit differs')
    engine=read(folder/'game-runtime-hashes.json')
    require(len(engine)==2 and len({(v['godot'],v['files'],v['project']) for v in engine.values()})==1,
            'Game assets or Godot builds differ between hosts')
    inputs={}
    for cohort in cohorts:
        for arm in GROUPS:
            d=prior if cohort=='original' and arm!='pure' else folder/cohort
            budget=read(d/'budget.json')
            require(budget['game_eval_seeds']==protocol['game_eval_seeds'] and
                    budget['training_seeds']==cohorts[cohort] and budget['train_steps']==96 and
                    budget['rollout_batch']==12 and budget['eval_horizon']==48 and
                    hashlib.sha256((d/'replay.json').read_bytes()).hexdigest()==protocol['source_sha'],
                    'Budget, holdout, or replay differs')
            _,part=inspect_arm(d,arm,budget)
            for seed,(r,reg,traces) in part.items():
                require(r['update']['reward']==('native_R1' if arm=='pure' else 'R1_plus_confirmed_binary_R2')
                        and r['update']['r2_enabled'] is (arm!='pure') and not r['update']['stimulation'],
                        'Final eight-rank update disagrees with registered reward semantics')
                train=read(d/arm/str(seed)/'training-trace.json')
                inputs[seed,arm]=(r,reg,traces,train)
    rows=[]
    trajectories={a:{k:[] for k in NAMES} for a in GROUPS}
    wet_initial=set()
    native_initial=set()
    for seed in seeds:
        native,oreg,sreg=(inputs[seed,a][1] for a in GROUPS)
        require(oreg['replay_digest']==sreg['replay_digest'] and oreg['test_pairs']==sreg['test_pairs']
                and all(i==j for i,j,k in oreg['train_pairs']) and all(i!=j for i,j,k in sreg['train_pairs'])
                and [r[::2] for r in oreg['train_pairs']]==[r[::2] for r in sreg['train_pairs']]
                and oreg['config']['normalization']==sreg['config']['normalization'],
                'Physical donor mapping or normalization differs')
        require(native['config']['training']==oreg['config']['training']==sreg['config']['training'],
                'Training hyperparameters differ')
        pure,ordered,shuffled=(inputs[seed,a][0] for a in GROUPS)
        require(ordered['initial']==shuffled['initial'] and all(
                ordered['initial'][k]==pure['initial'][k] for k in ('actor','base','reference')),
                'Starting policy or frozen reference differs')
        wet_initial.add(ordered['initial']['critic'])
        native_initial.add(pure['initial']['critic'])
        if seed in cohorts['added']:
            require(pure['initial']['critic']==ordered['native_critic_sha256']==shuffled['native_critic_sha256'],
                    'Baseline and MEA native critic weights differ')
        require(len({inputs[seed,a][3][0]['obs_hash'] for a in GROUPS})==1,'Training initial game state differs')
        for i,gs in enumerate(protocol['game_eval_seeds']):
            r=dict(training_seed=seed,game_seed=gs,cohort='added' if seed in cohorts['added'] else 'original',
                   health_path={})
            initial_game=set()
            for a in GROUPS:
                outcome,_,traces,_=inputs[seed,a]
                e=outcome['eval'][i]
                require(e['seed']==gs,'Game seed pairing differs')
                r[a]=e
                steps=traces[str(gs)]
                start_hp=steps[0]['before']['health']
                r['health_path'][a]=dict(initial=start_hp,
                        lowest=min([start_hp]+[s['after']['health'] for s in steps]))
                initial_game.add(steps[0]['obs_hash'])
                require(steps[-1]['terminal']==bool(e['death']),'Death flag differs')
                for k in NAMES:
                    if k=='explored':
                        curve=[s['after']['visited_count']-steps[0]['before']['visited_count'] for s in steps]
                    else:
                        curve=[s['after'][k] for s in steps]
                    require(curve[-1]==e[k],'Endpoint differs from measured trajectory')
                    trajectories[a][k].append(curve+[curve[-1]]*(48-len(curve)))
            require(len(initial_game)==1,'Held-out game initial observation differs')
            t1,t2=(inputs[seed,a][2][str(gs)] for a in ('ordered','shuffled'))
            require(all(s['context_donor']==t['context_donor'] for s,t in zip(t1,t2)),
                    'Wet arms evaluated different donor sequences')
            rows.append(r)
    require(len(wet_initial)==len(native_initial)==1,'Inconsistent warm critic across cohorts')
    summary={'all':summarize(rows,seeds)}
    for k,subset in cohorts.items():
        summary[k]=summarize([r for r in rows if r['training_seed'] in subset],subset)
    historical=read(prior/'comparison.json')['summary']
    for k in NAMES:
        for arm in ('ordered','shuffled'):
            require(abs(historical[k][arm]-summary['original']['means'][arm][k])<1e-9,
                    'Original paired results changed')
    result=dict(protocol=protocol,summary=summary,matched_results=rows,
                independent_chips=1,physical_files=9,train_physical_files=6,holdout_physical_files=3)
    (folder/'three-arm-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (folder/'paired-game-endpoints.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['cohort','training_seed','game_seed','arm',*METRICS])
        for r in rows:
            for a in GROUPS:
                writer.writerow([r['cohort'],r['training_seed'],r['game_seed'],a,*[r[a][k] for k in METRICS]])
    return result,trajectories


def plot(folder,result,trajectories):
    summary=result['summary']['all']
    with _plot_style():
        fig=Figure(figsize=(10.4,6.5),dpi=150,layout='constrained')
        fig.suptitle('22554 三组离线训练的留存游戏表现\n每条线平均 6 个训练种子 × 8 局游戏',fontsize=14)
        for ax,(k,name) in zip(fig.subplots(2,2).flat,NAMES.items()):
            for arm,label in GROUPS.items():
                arr=np.array(trajectories[arm][k])
                ax.plot(range(1,49),arr.mean(axis=0),label=label,color=COLORS[arm],ls=STYLES[arm],lw=2)
            ax.set(xlim=(1,48),xlabel='游戏决策步',ylabel=name)
            ax.grid(axis='y',alpha=.2)
            if k=='health' and all(np.ptp(trajectories[a][k])==0 for a in GROUPS):
                ax.set_ylim(98,101)
            if k=='explored':ax.legend(frameon=False,fontsize=9)
            ax.spines[['top','right']].set_visible(False)
        _save(fig,folder/'three-arm-trajectories.png')
        fig=Figure(figsize=(10.7,6.7),dpi=150,layout='constrained')
        fig.suptitle('有序 MEA 相对两个对照的终局差值\n每个点平均同一训练种子的 8 局；正值表示有序组更高',fontsize=14)
        seeds=summary['training_seeds']
        labels=[str(s)[-2:] for s in seeds]
        for ax,(k,name) in zip(fig.subplots(2,2).flat,NAMES.items()):
            ax.axhline(0,c='#849294',lw=.9)
            ax.axvline(2.5,c='#b5babc',lw=.8,ls=':')
            for other,offset,marker in [('pure',-.12,'o'),('shuffled',.12,'D')]:
                values=summary['effects']['ordered-'+other][k]
                ax.scatter(np.arange(len(seeds))+offset,values['by_seed'],s=43,marker=marker,color=COLORS[other],
                           label='有序减'+('纯模型' if other=='pure' else '打乱组'),zorder=3)
            ax.set(xticks=np.arange(len(seeds)),xticklabels=labels,xlabel='训练种子末两位（04–06 为新增）',
                   ylabel=name+'差值',xlim=(-.5,len(seeds)-.5))
            if k=='explored':ax.legend(frameon=False,fontsize=8)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.18)
        _save(fig,folder/'ordered-paired-differences.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder
    result,trajectories=analyze(folder)
    plot(folder,result,trajectories)
    for k,g in result['summary'].items():
        print(k,g['means'],{name:c['explored']['by_seed'] for name,c in g['effects'].items()})
