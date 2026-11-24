"""Audit eight-rank updates and quantify paired game and action differences."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from report_recorded_pulses import Figure, _plot_style, _save, read, np
from mea_state2_r2.protocol import BETA, GAMMA
from mea_state2_r2.reward_mix import calibrated_reward, discounted_ticks


ARMS = ('ordered', 'shuffled')
NAMES = {'explored': '探索新格数', 'health': '生命值', 'hunger': '饱食', 'thirst': '水分'}
COLORS = ('#17736b', '#aa4c21', '#375780')


def require(condition, description):
    if not condition:
        raise ValueError(description)


def check_recorded_reward(step, phase, config):
    sign = None if step['terminal'] else (1 if phase == 'pre' else -1)
    if 'reward_mix' in config:
        value, addition = calibrated_reward(step['r1'], sign, step['duration'], GAMMA, config['reward_mix'])
        require(step['r2_event_sign'] == sign and math.isclose(step['r2'], value, abs_tol=1e-14),
                'Calibrated reward or event sign differs')
    else:
        require(step['r2'] == sign, 'Recorded event sign differs')
        addition = 0. if sign is None else BETA * sign * discounted_ticks(step['duration'], GAMMA)
    require(math.isclose(step['r2_contribution'], addition, abs_tol=1e-14) and
            math.isclose(step['reward'], step['r1'] + addition, abs_tol=1e-14),
            'Recorded PPO reward differs')


def inspect_arm(folder, arm, budget):
    result = read(folder/arm/'result.json')
    pure = arm == 'pure'
    seeds = budget['training_seeds']
    horizon = budget['eval_horizon']
    require(result['arm'] == arm and result['budget'] == budget and
            [r['training_seed'] for r in result['seeds']] == seeds, 'Budget or training seeds differ')
    verified = {}
    for r in result['seeds']:
        seed = r['training_seed']
        d = folder/arm/str(seed)
        reg, train, updates, games = (read(d/n) for n in
                                     ('registration.json','training-trace.json','updates.json','eval-trace.json'))
        require(reg['config']['r2_enabled'] is (not pure) and not reg['config']['stimulation'] and
                reg['config']['devices'] == list(range(8)), 'Incorrect offline configuration')
        require(r['checkpoint']['reload_verified'] and r['checkpoint']['state'] == r['after'],
                'Checkpoint reload not verified')
        require(len(train) == r['training_steps'] == budget['train_steps'] and
                len(updates) == r['update_count'] == len(train)//budget['rollout_batch'], 'Missing training steps')
        prev = r['initial']
        for u in updates:
            require(u['world_size'] == 8 and u['unique_decisions'] == budget['rollout_batch'] and
                    all(u['checks'].values()) and u['actor_changed'] and u['critic_changed'] and
                    u['before'] == prev, 'Incomplete or discontinuous eight-rank update')
            prev = u['after']
        require(prev == r['after'], 'Updates and final checkpoint differ')
        indices = {phase: {x[0] for x in reg[phase+'_pairs']} for phase in ('train','test')}
        require(len(indices['train']) == (0 if pure else 6) and len(indices['test']) == (0 if pure else 3) and
                not indices['train'] & indices['test'], 'Train/holdout contamination')
        if pure:
            require(reg['physical_files']==0 and reg['config']['source']=='game' and
                    reg['replay_digest'] is None and r['initial']['critic']==r['native_critic_sha256'],
                    'Native model or no-signal registration differs')
            for trace in [train,*games.values()]:
                for s in trace:
                    require(s['r2'] is None and s['r2_contribution']==0 and s['reward']==s['r1']
                            and all(s[k] is None for k in ('context_donor','next_donor','r2_trial',
                                                           'next_state2_trial','recorded_phase'))
                            and s['actor_native_messages'] and s['critic_input_size']==108,
                            'MEA signal or auxiliary reward entered pure R1')
        for idx, step in enumerate(train):
            if pure:
                require(step['decision']==idx+1 and (idx==0 or
                        step['episode']==train[idx-1]['episode']+int(train[idx-1]['terminal'])),
                        'Native training trajectory discontinuity')
                continue
            ref = reg['train_pairs'][idx % len(reg['train_pairs'])]
            require([step['r2_trial'],step['next_state2_trial'],step['recorded_phase']] == ref and
                    step['decision'] == idx+1, 'Training donor mapping differs')
            check_recorded_reward(step, ref[2], reg['config'])
            if idx and not train[idx-1]['terminal']:
                require(train[idx-1]['next_donor'] == step['context_donor'] and
                        train[idx-1]['episode'] == step['episode'], 'Training reset at a nonterminal step')
        require([e['seed'] for e in r['eval']] == budget['game_eval_seeds'], 'Missing held-out games')
        for game in r['eval']:
            steps = games[str(game['seed'])]
            require(len(steps) == game['decisions'] <= horizon and
                    (len(steps) == horizon or steps[-1]['terminal']) and
                    not any(s['terminal'] for s in steps[:-1]), 'Game horizon or termination differs')
            require(game['explored'] == steps[-1]['after']['visited_count']-steps[0]['before']['visited_count']
                    and all(game[k] == steps[-1]['after'][k] for k in ('health','hunger','thirst')),
                    'Reported game metrics differ from step trace')
            for idx, s in enumerate(steps):
                if pure:
                    require(s['decision']==idx+1 and s['episode']==0 and
                            abs(sum(s['verb_probabilities'])-1)<.0001, 'Invalid pure R1 evaluation')
                    continue
                ref = reg['test_pairs'][idx % len(reg['test_pairs'])]
                check_recorded_reward(s, ref[2], reg['config'])
                require([s['r2_trial'],s['next_state2_trial'],s['recorded_phase']] == ref and
                        (idx == 0 or steps[idx-1]['next_donor'] == s['context_donor']) and
                        abs(sum(s['verb_probabilities'])-1) < 0.0001, 'Invalid evaluation trace')
        verified[seed] = (r, reg, games)
    return result, verified


def compare(folder):
    budget = read(folder/'budget.json')
    source = read(folder.parent/'raw-source-audit.json')
    require(hashlib.sha256((folder/'replay.json').read_bytes()).hexdigest() ==
            source['replay_sha256'] == budget['replay_sha256'], 'Replay checksum differs')
    data = {arm: inspect_arm(folder, arm, budget) for arm in ARMS}
    a,b = (data[arm][0] for arm in ARMS)
    require((a['study'], a['replay_digest']) == (b['study'], b['replay_digest']), 'Physical source differs')
    pairs, actions = [], []
    seeds, horizon = budget['training_seeds'], budget['eval_horizon']
    for seed in seeds:
        (a, ra, ta), (b, rb, tb) = (data[arm][1][seed] for arm in ARMS)
        require(a['initial'] == b['initial'] and ra['test_pairs'] == rb['test_pairs'] and
                ra['config']['normalization'] == rb['config']['normalization'], 'Unmatched groups')
        require(all(i==j for i,j,_ in ra['train_pairs']) and all(i!=j for i,j,_ in rb['train_pairs'])
                and [x[::2] for x in ra['train_pairs']] == [x[::2] for x in rb['train_pairs']],
                'Expected phase-preserving signal shuffle missing')
        for x,y in zip(a['eval'],b['eval']):
            gs = x['seed']
            p = dict(training_seed=seed, game_seed=gs, ordered=x, shuffled=y)
            p['delta'] = {k:x[k]-y[k] for k in (*NAMES,'death','r1','decisions','ticks')}
            traces = [ta[str(gs)],tb[str(gs)]]
            p['exploration_curve'] = [[s['after']['visited_count']-steps[0]['before']['visited_count']
                                      for s in steps] for steps in traces]
            for curve in p['exploration_curve']:
                curve.extend([curve[-1]]*(horizon-len(curve)))
            first = None
            for idx, (u,v) in enumerate(zip(*traces), start=1):
                require(u['context_donor'] == v['context_donor'], 'Evaluation MEA donors differ')
                same_action = u['action'] == v['action']
                if not same_action and first is None:
                    first = idx
                if u['obs_hash'] == v['obs_hash']:
                    tv = sum(abs(i-j) for i,j in zip(u['verb_probabilities'],v['verb_probabilities']))/2
                    actions.append(dict(training_seed=seed,game_seed=gs,step=idx,verb_tv=tv,
                                        same_action=same_action, selected_prob_delta=(
                                        abs(math.exp(u['logp'])-math.exp(v['logp'])) if same_action else None)))
            require(first is not None or x['decisions'] == y['decisions'], 'Game lengths differ with same actions')
            p['first_different_action'] = first
            pairs.append(p)
    summary = {}
    for k in pairs[0]['delta']:
        diffs = [p['delta'][k] for p in pairs]
        summary[k] = dict(ordered=mean(p['ordered'][k] for p in pairs),
                          shuffled=mean(p['shuffled'][k] for p in pairs), difference=mean(diffs),
                          by_seed=[mean(p['delta'][k] for p in pairs if p['training_seed']==s) for s in seeds],
                          better_equal_worse=[sum(d>1e-9 for d in diffs),sum(abs(d)<=1e-9 for d in diffs),
                                              sum(d< -1e-9 for d in diffs)])
    require(actions, 'No common game states')
    tvs = [d['verb_tv'] for d in actions]
    probs = [d['selected_prob_delta'] for d in actions if d['same_action']]
    diag = dict(matched_decisions=len(actions), common_state_action_mismatches=sum(not d['same_action'] for d in actions),
                positive_verb_tv=sum(v>0 for v in tvs), mean_verb_tv=mean(tvs), max_verb_tv=max(tvs),
                same_action_n=len(probs), mean_abs_selected_prob_delta=mean(probs) if probs else None,
                games_with_action_divergence=sum(p['first_different_action'] is not None for p in pairs),
                total_overlapping_decisions=sum(min(p[a]['decisions'] for a in ARMS) for p in pairs))
    seed_table=[]
    for seed in seeds:
        group=[p for p in pairs if p['training_seed']==seed]
        votes=[p['delta']['explored'] for p in group]
        ac=[d for d in actions if d['training_seed']==seed]
        seed_table.append(dict(training_seed=seed,paired_games=len(group),
                endpoint_means={a:{k:mean(p[a][k] for p in group) for k in NAMES} for a in ARMS},
                tile_wins=sum(x>0 for x in votes),tile_ties=votes.count(0),tile_losses=sum(x<0 for x in votes),
                common_observation_steps=len(ac),mean_verb_tv=mean(d['verb_tv'] for d in ac),
                games_with_different_actions=sum(p['first_different_action'] is not None for p in group)))
    short=read(folder.parent/'paired-game-comparison.json')
    same_horizon={}
    for idx,arm in enumerate(ARMS):
        same_horizon[arm]=mean(p['exploration_curve'][idx][5] for p in pairs)
    paired_seed_data=[(p['training_seed'],p['game_seed']) for p in pairs]
    require(paired_seed_data==[(p['training_seed'],p['game_seed']) for p in short['pairs']],
            'Historical game pair IDs do not match')
    horizons=dict(previous_12_train_6_eval={arm:short['summary']['explored'][arm] for arm in ARMS},
                  current_96_train_at_6=same_horizon,
                  current_96_train_at_48={arm:summary['explored'][arm] for arm in ARMS})
    result = dict(summary=summary, action_diagnostics=diag, pairs=pairs, seed_table=seed_table, stable_exploration=all(
                  d>0 for d in summary['explored']['by_seed']), independent_chip_count=1, holdout_physical_trials=3,
                  exploration_by_horizon=horizons,
                  completed_to_horizon={a:sum(p[a]['decisions']==horizon for p in pairs) for a in ARMS})
    (folder/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (folder/'matched-action-probabilities.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=actions[0].keys())
        w.writeheader()
        w.writerows(actions)
    return budget, result, actions


def draw(folder, budget, result, actions):
    pairs, summary = result['pairs'], result['summary']
    horizon, seeds = budget['eval_horizon'], budget['training_seeds']
    steps = np.arange(1,horizon+1)
    with _plot_style():
        fig = Figure(figsize=(10.5,5.1),dpi=145,layout='constrained')
        fig.suptitle('22554 有序与打乱信号   ·   留存游戏的探索轨迹',fontsize=16)
        left,right = fig.subplots(1,2)
        for idx,(name,style) in enumerate([('有序信号','-'),('打乱信号','--')]):
            curves = np.array([p['exploration_curve'][idx] for p in pairs])
            left.plot(steps,curves.mean(axis=0),style,color=COLORS[idx],lw=2.1,label=name)
        for i,seed in enumerate(seeds):
            curves = np.array([np.subtract(*p['exploration_curve']) for p in pairs if p['training_seed']==seed])
            right.plot(steps,curves.mean(axis=0),color=COLORS[i],lw=1.8,ls=['-','--',':'][i],label=str(seed))
        left.set(ylabel='平均累计探索新格数',title='每组 24 局')
        right.axhline(0,c='#777777',lw=.65,zorder=0)
        right.set(ylabel='有序减打乱（格）',title='各训练种子，分别平均 8 局')
        for ax in (left,right):
            ax.set(xlabel='游戏决策步',xlim=(1,horizon))
            ax.grid(axis='y',alpha=.20)
            ax.legend(frameon=False)
            ax.spines[['top','right']].set_visible(False)
        _save(fig,folder/'exploration-trajectory.png')
        fig = Figure(figsize=(10.4,6),dpi=145,layout='constrained')
        fig.suptitle('22554 配对差值   ·   有序信号减打乱信号',fontsize=16)
        for ax,(metric,title) in zip(fig.subplots(2,2).flat,NAMES.items()):
            offsets=np.linspace(-.20,.20,len(budget['game_eval_seeds']))
            ax.axhline(0,ls='--',color='#98a2a7',lw=.8)
            for i,seed in enumerate(seeds):
                diffs=[p['delta'][metric] for p in pairs if p['training_seed']==seed]
                ax.scatter(i+offsets,diffs,s=16,c=COLORS[i],alpha=.45)
                ax.plot([i-.22,i+.22],[mean(diffs)]*2,c=COLORS[i],lw=2.4)
            ax.set(xticks=range(len(seeds)),xticklabels=[str(s) for s in seeds],ylabel='每局配对差值',
                   title=f"{title}    平均差 {summary[metric]['difference']:+.3f}")
            ax.set_xlim(-.45,len(seeds)-.55)
            ax.spines[['top','right']].set_visible(False)
        _save(fig,folder/'paired-reserves-exploration.png')
        fig = Figure(figsize=(9.7,3.8),dpi=145,layout='constrained')
        fig.suptitle('同一游戏状态下的动作概率，以及两组首次选出不同动作的时间',fontsize=14)
        ax,bx=fig.subplots(1,2)
        for i,seed in enumerate(seeds):
            groups=[[r['verb_tv'] for r in actions if r['training_seed']==seed and r['step']==j] for j in steps]
            ax.plot(steps,[mean(g)*100 if g else np.nan for g in groups],c=COLORS[i],
                    ls=['-','--',':'][i],label=str(seed),lw=1.6)
        ax.set(ylabel='动作类别分布总变差（%）',xlabel='游戏决策步')
        ax.legend(frameon=False,fontsize=8)
        divs=[p['first_different_action'] for p in pairs]
        bx.step(steps,[sum(d is not None and d<=s for d in divs) for s in steps],where='post',color=COLORS[0])
        bx.set(xlabel='游戏决策步',ylabel='曾出现不同动作的配对局数',ylim=(-.5,len(pairs)+.5))
        for ax in (ax,bx):
            ax.set_xlim(1,horizon)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.2)
        _save(fig,folder/'action-probability.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder
    budget,result,actions=compare(folder)
    draw(folder,budget,result,actions)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=True))
