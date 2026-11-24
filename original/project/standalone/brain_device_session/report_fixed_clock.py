"""Paired fixed-tick outcomes for the ordered state2 x recorded R2 factorial experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from report_long_recorded import Figure, _plot_style, _save, check_recorded_reward, np, read, require
from mea_state2_r2.features import transform

ARMS={'pure':'纯模型 R1','stateonly':'仅有序 state2','r2only':'仅 R2',
      'ordered':'有序 state2 + R2','shuffled':'打乱 state2 + R2'}
METRICS={'explored':'探索新格数','health':'生命值','hunger':'饱食','thirst':'水分'}
COLORS={'pure':'#474f69','stateonly':'#248447','r2only':'#9770ac','ordered':'#117b87','shuffled':'#be682f'}
STYLES={'pure':'-','stateonly':'--','r2only':'-.','ordered':'-','shuffled':':'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_arm(folder,cohort,arm):
    prior=folder.parent
    return (prior if cohort=='original' and arm!='pure' else prior/'three-arm-six-seeds'/cohort)/arm


def audit_cohort(folder,cohort,plan):
    d=folder/cohort
    budget=read(d/'budget.json')
    require(sha(d/'replay.json')==plan['source_sha']==budget['replay_sha256'],'Wrong raw recording replay')
    require(budget['training_seeds']==plan['cohorts'][cohort] and
            budget['game_eval_seeds']==plan['game_eval_seeds'] and
            budget['train_steps']==96 and budget['rollout_batch']==12,'Unmatched training or evaluation budget')
    data=read(d/'replay.json')
    prime=read(d/'preceding-quiet.json')['prime']
    donors={w['id']:w for t in data['trials'] for w in (t['pre'],t['post'])}
    donors[prime['id']]=prime
    windows={w['id']:list(transform(w['features'],data['normalization'])) for w in donors.values()}
    for arm in ('stateonly','r2only'):
        state=arm=='stateonly'
        result=read(d/arm/'result.json')
        parent=prior_arm(folder,cohort,'ordered' if state else 'pure')
        ref={x['training_seed']:x for x in read(parent/'result.json')['seeds']}
        require(result['arm']==arm and result['budget']==budget and
                [x['training_seed'] for x in result['seeds']]==budget['training_seeds'],'Missing runs')
        for run in result['seeds']:
            seed=run['training_seed']
            p=d/arm/str(seed)
            reg=read(p/'registration.json')
            cfg=reg['config']
            register_order=read(prior_arm(folder,cohort,'ordered')/str(seed)/'registration.json')
            require(run['initial']==ref[seed]['initial'] and run['checkpoint']['state']==run['after']
                    and run['checkpoint']['reload_verified'] and run['checkpoint']['optimizer_roundtrip_verified'],
                    'Mismatched starting weights or checkpoint')
            require(cfg['r2_enabled'] is (not state) and cfg['stimulation'] is False and reg['physical_files']==9
                    and cfg['source']==('recorded' if state else 'game') and cfg['devices']==list(range(8))
                    and cfg['training']==register_order['config']['training'],'Arm configuration changed')
            require(reg['replay_digest']==register_order['replay_digest'] and
                    (not state or cfg['normalization']==data['normalization']),'Changed recording or normalizer')
            updates=read(p/'updates.json')
            require(len(updates)==run['update_count']==8 and run['training_steps']==96,'Incomplete training')
            previous=run['initial']
            for u in updates:
                require(u['world_size']==8 and u['unique_decisions']==12 and u['before']==previous
                        and u['r2_enabled'] is (not state) and all(u['checks'].values()) and
                        u['actor_changed'] and u['critic_changed'],'Model update disagreement')
                previous=u['after']
            require(previous==run['after'],'Final weights differ')
            games=read(p/'eval-trace.json')
            original_games=read(parent/str(seed)/'eval-trace.json')
            require([e['seed'] for e in run['eval']]==plan['game_eval_seeds'],'Eval seeds changed')
            for g in games:
                require(games[g][0]['obs_hash']==original_games[g][0]['obs_hash'],'Game initial state differs')
            train=read(p/'training-trace.json')
            require(len(train)==96,'Training rows missing')
            for split,trace in [('train',train)]+[('test',t) for t in games.values()]:
                order=register_order[split+'_pairs']
                require(reg[split+'_pairs']==[[i,j if state else None,phase] for i,j,phase in order],
                        'Wrong split or label mapping')
                prior=prime['id']
                for idx,s in enumerate(trace):
                    i,j,phase=order[idx%len(order)]
                    require(s['decision']==idx+1 and s['recorded_phase']==phase,'Invalid clock or label phase')
                    if state:
                        nxt=data['trials'][j][phase]['id']
                        require(s['r2'] is None and s['r2_trial'] is None and s['r2_contribution']==0 and
                                s['reward']==s['r1'] and s['context_donor']==prior and s['next_donor']==nxt
                                and s['next_state2_trial']==j and s['state2']==windows[prior] and
                                s['critic_input_size']==112,'state2-only label, reward or donor differs')
                        prior=prime['id'] if s['terminal'] else nxt
                    else:
                        check_recorded_reward(s,phase,cfg)
                        require(s['r2_trial']==i and
                                all(s[k] is None for k in ('context_donor','next_donor','next_state2_trial'))
                                and s['actor_native_messages'] and s['critic_input_size']==108,
                                'R2-only label, reward or model input differs')


def load_clock(folder,plan):
    records={a:{} for a in ARMS}
    for part in ('prior','original','added'):
        path=folder/f'clock-{part}.json'
        part_data=read(path)
        audit=read(folder/f'clock-{part}-audit.json')
        expected_engine=read(folder.parent/'three-arm-six-seeds/game-runtime-hashes.json')['192.0.2.23']
        require(sha(path)==audit['file_sha'] and part_data['engine']==expected_engine and
                part_data['source_sha256']==audit['source_sha'] and part_data['tick_horizon']==plan['tick_horizon'],
                'Clock audit failed')
        for a,seeds in part_data['arms'].items():
            require(not (set(seeds)&set(records[a])),'Repeated seed')
            records[a].update(seeds)
    return records


def effect_summary(arr):
    out={}
    for j,k in enumerate(METRICS):
        byseed=arr[:,:,j].mean(axis=1)
        out[k]=dict(mean=float(byseed.mean()),by_seed=byseed.tolist(),
                    higher_equal_lower=[int(sum(byseed>1e-8)),int(sum(abs(byseed)<=1e-8)),int(sum(byseed< -1e-8))],
                    paired_games=[int((arr[:,:,j]>1e-8).sum()),int((np.abs(arr[:,:,j])<=1e-8).sum()),
                                  int((arr[:,:,j]< -1e-8).sum())])
    return out


def analyze(folder,verified_records=None,trace_folders=None):
    plan=read(folder/'protocol.json')
    if verified_records is None:
        for c in plan['cohorts']:audit_cohort(folder,c,plan)
        records=load_clock(folder,plan)
    else:
        records=verified_records
    seeds=plan['cohorts']['original']+plan['cohorts']['added']
    game_seeds=plan['game_eval_seeds']
    horizon=plan['tick_horizon']
    curves={}
    means={}
    clock_audit={}
    drinking={}
    with (folder/'paired-tick48.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(('training_seed','game_seed','arm','elapsed_tick',*METRICS,'decision_covering_tick',
                    'actual_record_length_ticks','terminal_tick'))
        for a in ARMS:
            d=records[a]
            require(set(map(int,d))==set(seeds) and all(set(map(int,g))==set(game_seeds) for g in d.values()),
                    'Incomplete seed pairing')
            curves[a]=np.array([[d[str(s)][str(g)]['series'] for g in game_seeds] for s in seeds],dtype=float)
            require(curves[a].shape==(6,8,horizon+1,4) and np.isfinite(curves[a]).all(),'Missing simulation ticks')
            drink_counts=[]
            for s in seeds:
                cohort='original' if s in plan['cohorts']['original'] else 'added'
                if trace_folders is not None:
                    src=trace_folders[cohort,a]
                else:
                    src=prior_arm(folder,cohort,a) if a in ('pure','ordered','shuffled') else folder/cohort/a
                decisions=read(src/str(s)/'eval-trace.json')
                for g in game_seeds:
                    r=d[str(s)][str(g)]
                    w.writerow((s,g,a,horizon,*r['series'][horizon],r['decision_at_horizon'],r['actual_ticks'],r['terminal_tick']))
                    end=r['decision_at_horizon'] or r['verified_decisions']
                    drink_counts.append(sum(step['action']['verb']=='drink' for step in decisions[str(g)][:end]))
            drinking[a]=dict(drink_actions=sum(drink_counts),games_with_drink=sum(n>0 for n in drink_counts),
                             games=len(drink_counts))
            flat=[r for games in d.values() for r in games.values()]
            clock_audit[a]=dict(games=len(flat),decisions_verified=sum(r['verified_decisions'] for r in flat),
                                within_action=sum(r['decision_at_horizon'] is not None and not r['horizon_on_decision_boundary'] for r in flat),
                                deaths_by_horizon=sum(r['terminal_tick'] is not None and r['terminal_tick']<=horizon for r in flat),
                                minimum_actual_ticks=min(r['actual_ticks'] for r in flat))
            means[a]=dict(zip(METRICS,curves[a][:,:,horizon,:].mean(axis=(0,1)).tolist()))
    for x in curves.values():require(np.array_equal(x[:,:,0],curves['pure'][:,:,0]),'Initial metrics mismatch')
    p,s,r,j,h=(curves[a][:,:,-1,:] for a in ARMS)
    contrasts={'state_vs_pure':s-p,'r2_vs_pure':r-p,'state_given_r2':j-r,'r2_given_state':j-s,
               'interaction':(j-r)-(s-p),'joint_vs_pure':j-p,'ordered_vs_shuffle':j-h,'shuffle_vs_pure':h-p,
               'factorial_state_mean':.5*((s-p)+(j-r)),'factorial_r2_mean':.5*((r-p)+(j-s))}
    require(np.allclose(contrasts['joint_vs_pure'],contrasts['state_vs_pure']+contrasts['r2_vs_pure']+
                        contrasts['interaction']),'Factorial identity failed')
    effects={k:effect_summary(v) for k,v in contrasts.items()}
    result=dict(tick_horizon=horizon,training_seeds=seeds,game_seeds=game_seeds,means=means,effects=effects,
                initial_means=dict(zip(METRICS,curves['pure'][:,:,0,:].mean(axis=(0,1)).tolist())),
                drinking_actions_to_horizon=drinking,
                per_arm_seed_means={a:[dict(zip(METRICS,v.tolist())) for v in x[:,:,-1].mean(axis=1)] for a,x in curves.items()},
                clock_audit=clock_audit,independent_chips=1,physical_trains=6,physical_holdouts=3)
    if 'reward_mix' in plan:result['reward_mix']=plan['reward_mix']
    (folder/'factorial-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return curves,result


def plot(folder,curves,result):
    with _plot_style():
        fig=Figure(figsize=(11.4,6.9),dpi=145,layout='constrained')
        heading='22554 校准 R2 后的五组对照' if 'reward_mix' in result else '22554 五组对照'
        fig.suptitle(heading+' ｜ 同一游戏时钟刻的实测轨迹\n每组 6 个训练种子 × 8 局，曲线从重置时刻开始',fontsize=14)
        for j,(ax,(k,title)) in enumerate(zip(fig.subplots(2,2).flat,METRICS.items())):
            for a,label in ARMS.items():
                ax.plot(range(result['tick_horizon']+1),curves[a][:,:,:,j].mean(axis=(0,1)),
                        label=label,c=COLORS[a],ls=STYLES[a],lw=1.9)
            ax.set(xlim=(0,result['tick_horizon']),xlabel='经过的游戏时钟刻',ylabel=title)
            if k=='health' and all(np.ptp(t[:,:,:,j])==0 for t in curves.values()):ax.set_ylim(99,101)
            if k=='explored':ax.legend(frameon=False,fontsize=8,loc='upper left')
            ax.grid(axis='y',alpha=.18)
            ax.spines[['top','right']].set_visible(False)
        _save(fig,folder/'five-arm-clock-trajectories.png')
        fig=Figure(figsize=(11.4,7.0),dpi=145,layout='constrained')
        fig.suptitle('第 48 个游戏时钟刻 ｜ 单独加入与联用的效果\n小点为各训练种子的 8 局平均差，菱形为 6 个种子的平均差',fontsize=14)
        contrasts={'state_vs_pure':'单加 state2','r2_vs_pure':'单加 R2','interaction':'交互效应','joint_vs_pure':'两者联用'}
        if 'reward_mix' in result:
            contrasts['ordered_vs_shuffle']='有序减打乱'
        positions=list(reversed(range(len(contrasts))))
        for ax,(metric,title) in zip(fig.subplots(2,2).flat,METRICS.items()):
            ax.axvline(0,c='#8a939e',lw=1)
            for y,contrast in zip(positions,contrasts):
                vals=result['effects'][contrast][metric]
                ax.scatter(vals['by_seed'],y+np.linspace(-.15,.15,6),s=16,color='#639c9a',alpha=.85)
                ax.scatter(vals['mean'],y,marker='D',s=43,color='#273f60',zorder=3)
            ax.set(yticks=positions,yticklabels=list(contrasts.values()),ylim=(-.5,len(contrasts)-.5),xlabel=title+'差值')
            ax.grid(axis='x',alpha=.16)
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(axis='y',length=0)
        _save(fig,folder/'fixed-clock-factor-effects.png')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    folder=p.parse_args().folder
    curves,result=analyze(folder)
    plot(folder,curves,result)
    print(json.dumps(dict(means=result['means'],clock_audit=result['clock_audit'])))
