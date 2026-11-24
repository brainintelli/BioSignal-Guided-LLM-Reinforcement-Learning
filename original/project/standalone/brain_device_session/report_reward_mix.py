"""Audit real reward inputs and compare both policies at the same native game tick."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from report_fixed_clock import (ARMS, METRICS, Figure, _plot_style, _save, analyze,
                               load_clock, np, plot, prior_arm, read, require, sha)


def stats(rows):
    ratios=[abs(r['r2_contribution']/r['r1']) for r in rows if r['r1']!=0]
    return dict(steps=len(rows),sum_abs_ratio=sum(abs(r['r2_contribution']) for r in rows)/sum(abs(r['r1']) for r in rows),
                max_step_ratio=max(ratios),median_step_ratio=statistics.median(ratios),
                zero_r1_steps=sum(r['r1']==0 for r in rows),
                dominant_steps=sum(abs(r['r2_contribution'])>abs(r['r1']) for r in rows),
                sign_flips=sum(r['reward']*r['r1']<0 for r in rows))


def audit(folder,plan,paths):
    prior=folder.parent/'factorial-tick48'
    mix=plan['reward_mix']
    calibration=read(folder/'calibration.json')
    require(calibration['mix']==mix,'Unregistered scale')
    rates=[]
    for co,ss in plan['cohorts'].items():
        for s in ss:
            path=paths[co,'pure']/str(s)/'training-trace.json'
            require(sha(path)==calibration['training_trace_hashes'][str(s)],'Reference data changed')
            for step in read(path):
                require(step['r1']==step['reward'] and not step['r2_contribution'],'Calibration used non-native rewards')
                if step['r1'] and not step['terminal']:
                    rates.append(abs(step['r1'])/sum(.99985**i for i in range(step['duration'])))
    require(len(rates)==calibration['valid_reference_steps'] and math.isclose(statistics.median(rates),mix['reference_per_tick'],rel_tol=1e-12),
            'Recomputed reference differs')
    totals={a:{'train':[],'eval':[]} for a in plan['rerun_arms']}
    seed_stats={}
    for co,seeds in plan['cohorts'].items():
        d=folder/co
        budget=read(d/'budget.json')
        data=read(d/'replay.json');prime=read(d/'preceding-quiet.json')['prime']['id']
        require(budget['training_seeds']==seeds and budget['reward_mix']==mix and sha(d/'replay.json')==plan['source_sha'],
                'Training registration differs')
        for arm in plan['rerun_arms']:
            newdir=paths[co,arm]
            olddir=prior/co/arm if arm=='r2only' else prior_arm(folder,co,arm)
            result=read(newdir/'result.json')
            previous={r['training_seed']:r for r in read(olddir/'result.json')['seeds']}
            require(result['budget']==budget and [r['training_seed'] for r in result['seeds']]==seeds,'Missing model seeds')
            for run in result['seeds']:
                s=run['training_seed'];p=newdir/str(s);old=olddir/str(s)
                a,b=read(p/'registration.json'),read(old/'registration.json')
                cfg=dict(a['config'])
                require(cfg.pop('reward_mix')==mix and a['config']['r2_enabled'] and not a['config']['stimulation']
                        and all(cfg[k]==b['config'][k] for k in cfg if k not in ('model_paths','project','godot'))
                        and all(a[k]==b[k] for k in ('physical_files','replay_digest','train_pairs','test_pairs','eval_seeds')),
                        'New and previous runs differ on an experimental condition')
                require(run['initial']==previous[s]['initial'] and run['checkpoint']['state']==run['after']
                        and run['checkpoint']['reload_verified'] and run['checkpoint']['optimizer_roundtrip_verified']
                        and run['training_steps']==96 and run['update_count']==8,'Missing model equivalence or checkpoint verification')
                state=run['initial']
                updates=read(p/'updates.json')
                require(len(updates)==8,'Missing PPO updates')
                for update in updates:
                    require(update['before']==state and update['r2_enabled'] and update['world_size']==8 and
                            update['unique_decisions']==12 and update['actor_changed'] and update['critic_changed']
                            and all(update['checks'].values()),'Incomplete eight-rank PPO update')
                    state=update['after']
                require(state==run['after'],'Final checkpoint changed')
                train=read(p/'training-trace.json');ev=read(p/'eval-trace.json')
                require(len(train)==96 and [g['seed'] for g in run['eval']]==plan['game_eval_seeds'] and
                        set(map(int,ev))==set(plan['game_eval_seeds']),'Evaluation set changed')
                for split,trace in [('train',train)]+[('eval',rows) for rows in ev.values()]:
                    order=a['train_pairs' if split=='train' else 'test_pairs']
                    context=prime
                    for idx,r in enumerate(trace):
                        source,donor,phase=order[idx%len(order)]
                        sgn=None if r['terminal'] else (1 if phase=='pre' else -1)
                        dt=sum(.99985**j for j in range(r['duration']))
                        expected=0. if sgn is None else sgn*mix['amplitude_fraction']*min(abs(r['r1']),mix['reference_per_tick']*dt)
                        inc=mix['r2_weight']*expected
                        require(r['r2_event_sign']==sgn and r['r2_trial']==source and r['recorded_phase']==phase
                                and r['next_state2_trial']==donor and math.isclose(r['r2'],expected,abs_tol=1e-14)
                                and math.isclose(r['r2_contribution'],inc,abs_tol=1e-14)
                                and math.isclose(r['reward'],mix['r1_weight']*r['r1']+inc,abs_tol=1e-14)
                                and abs(inc)<=mix['amplitude_fraction']*mix['r2_weight']*abs(r['r1'])+1e-14,
                                'Reward formula or direction differs')
                        if arm=='r2only':
                            require(r['context_donor'] is None and r['next_donor'] is None and r['actor_native_messages']
                                    and r['critic_input_size']==108,'Signal entered reward-only model')
                        else:
                            nxt=data['trials'][donor][phase]['id']
                            require(r['context_donor']==context and r['next_donor']==nxt,'Measured input sequence changed')
                            context=prime if r['terminal'] else nxt
                    totals[arm][split].extend(trace)
                seed_stats[arm+'-'+str(s)]=stats(train)
    return dict(by_arm={a:{k:stats(rows) for k,rows in sp.items()} for a,sp in totals.items()},
                per_seed_training=seed_stats,legacy_training=calibration['legacy_training_audit'])


def run(folder):
    plan=read(folder/'protocol.json');prior=folder.parent/'factorial-tick48'
    paths={}
    for co in plan['cohorts']:
        for a in ARMS:
            paths[co,a]=folder/co/a if a in plan['rerun_arms'] else prior/co/a if a=='stateonly' else prior_arm(folder,co,a)
    rewards=audit(folder,plan,paths)
    previous_clock=load_clock(prior,read(prior/'protocol.json'))
    records={a:previous_clock[a] if a in plan['unchanged_controls'] else {} for a in ARMS}
    for co,seeds in plan['cohorts'].items():
        path=folder/f'clock-{co}.json'
        c=read(path);meta=read(folder/f'clock-{co}-audit.json')
        require(sha(path)==meta['file_sha'] and c['source_sha256']==meta['source_sha'] and c['tick_horizon']==48
                and c['engine']==read(folder.parent/'three-arm-six-seeds/game-runtime-hashes.json')['192.0.2.23']
                and set(c['arms'])==set(plan['rerun_arms']),'Clock replay mismatch')
        for a,games in c['arms'].items():
            require(set(map(int,games))==set(seeds),'Missing clock replay')
            records[a].update(games)
    curves,result=analyze(folder,records,paths)
    old=read(prior/'factorial-results.json')
    rewards['change_in_endpoints']={a:{k:result['means'][a][k]-old['means'][a][k] for k in METRICS} for a in ARMS}
    rewards['old_means']=old['means'];rewards['new_means']=result['means']
    (folder/'reward-audit.json').write_text(json.dumps(rewards,indent=2,allow_nan=False),encoding='utf8')
    plot(folder,curves,result)
    with _plot_style():
        fig=Figure(figsize=(10.2,4.0),dpi=150,layout='constrained')
        fig.suptitle('22554 奖励校准 ｜ 实际进入 PPO 的 R2 加权贡献',fontsize=14)
        arms=['r2only','ordered','shuffled'];x=np.arange(3)
        for ax,key,title in zip(fig.subplots(1,2),('sum_abs_ratio','max_step_ratio'),
                                ('全部训练步的绝对贡献之和比','单步贡献比的最大值')):
            for dx,field,label,col in [(-.18,'legacy_training','旧设定','#77849b'),(.18,'by_arm','校准后','#157d75')]:
                vals=[100*(rewards[field][a] if field=='legacy_training' else rewards[field][a]['train'])[key] for a in arms]
                bars=ax.bar(x+dx,vals,.35,label=label,color=col)
                ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
            ax.set(xticks=x,xticklabels=['仅 R2','有序 + R2','打乱 + R2'],ylabel='|加权 R2| / |R1|（%）',title=title)
            ax.margins(y=.18);ax.legend(frameon=False,fontsize=8);ax.spines[['right','top']].set_visible(False)
        _save(fig,folder/'actual-reward-ratios.png')
    print(json.dumps({'means':result['means'],'reward_ratios':rewards['by_arm']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    run(p.parse_args().folder)
