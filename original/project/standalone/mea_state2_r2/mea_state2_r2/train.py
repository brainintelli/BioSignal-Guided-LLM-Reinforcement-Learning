"""Actual on-policy game training with historical observations and simulated R2."""
import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import observation_key
from agent.mea_long_ablation import ORIGINAL_TENSORS
from tools.mea_offline import evidence, _value, policy_statistics, validate_policy_statistics
from .data import Dataset
from .features import digest, transition_context
from .protocol import PROTOCOL, GAMMA, Rewards, cell
from .trajectory import finish, audit


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False, allow_nan=False), flush=True)


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def prepare(policy, data, obs, position, arm, block, split='train', phase='after'):
    if arm == 'A0':
        return None, policy.prepare([obs])[0]
    snapshot = data.snapshot(position, phase, observation_key(obs), arm, block, split)
    return snapshot, policy.prepare([obs], [snapshot])[0]


def stored(row):
    return row | {'prepared': evidence(row['prepared'])}


def collect(policy, data, env, obs, tracker, rewards, seed, arm, block, start, count, output):
    import torch
    schedule = cell(block, arm)
    rows = []
    for index in range(start, start + count):
        snapshot, prepared = prepare(policy, data, obs, index, arm, block)
        torch.manual_seed(schedule['action_seed'] + index)
        sampled = policy.act([prepared], with_reference=True)
        action = sampled['actions'][0]
        nxt = env.step(action)
        score = tracker.score(nxt['transition'], nxt['observation'])
        p = nxt['observation']['player']
        reward = rewards.score(score['reward'], score['elapsed_ticks'], nxt['terminated'],
                   dict(H=p['health'], F=p['hunger'], W=p['thirst']),
                   transition_context(nxt['observation'],nxt['terminated'],observation_key))
        next_snapshot = None if nxt['terminated'] else data.snapshot(index+1,'after',
                         observation_key(nxt['observation']),arm,block,'train')
        row = dict(index=index, seed=seed, observation=obs, snapshot=snapshot, prepared=prepared,
            action=action, action_source='model', transition=nxt['transition'], next_observation=nxt['observation'],
            next_snapshot=next_snapshot, terminal=nxt['terminated'], r1_score=score, reward_record=reward,
            reward=reward['total_reward'], duration=score['elapsed_ticks'], value=sampled['values'][0])
        for key, plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                            ('reference_verb','reference_verb'),('reference_target','reference_target')]:
            row[key] = sampled[plural][0]
        obs = nxt['observation']
        if row['terminal']:
            seed += 1
            obs = env.reset(seed)['observation']
            tracker = Reward(obs, 'bottleneck', GAMMA)
            row['reset'] = dict(seed=seed, observation=obs)
        rows.append(row)
        append(output/'collection.partial.jsonl', stored(row))
        if (index+1) % 32 == 0:
            emit('collected', block=block, arm=arm, decisions=index+1)
    if rows[-1]['terminal']:
        bootstrap = dict(terminal=True, value=0., snapshot=None, prepared=None)
    else:
        snap, prepared = prepare(policy,data,obs,start+count,arm,block)
        bootstrap = dict(terminal=False,value=_value(policy,prepared),snapshot=snap,prepared=evidence(prepared))
    finish(rows,bootstrap['value'],GAMMA,.999)
    return rows,bootstrap,obs,tracker,seed


def metrics(rows, horizon=150):
    if not rows:
        raise ValueError('No evaluation decisions')
    rows = rows[:horizon]
    last = rows[-1]['transition']['after']
    r1 = math.fsum(r['r1_score']['reward'] for r in rows)
    ticks = sum(r['duration'] for r in rows)
    return dict(r1_sum=r1, r1_per_tick=r1/ticks, visited_count=last['visited_count'],
                final_health=last['health'], final_hunger=last['hunger'], final_thirst=last['thirst'],
                elapsed_ticks=ticks, terminated=int(rows[-1]['terminal']), decisions=len(rows),
                decision_budget=horizon, absorbed_slots=horizon-len(rows) if rows[-1]['terminal'] else 0)


def evaluate_episode(policy,data,args,phase,out):
    import torch
    schedule=cell(args.block,args.arm)
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with GodotEnv(str(args.godot),str(args.project),map_mode='infinite_freeze',navigation_steps=4) as env:
        obs=env.reset(schedule['eval_world_seed'])['observation']
        tracker=Reward(obs,'bottleneck',GAMMA)
        for index in range(PROTOCOL['eval_decisions']):
            snapshot, prepared=prepare(policy,data,obs,index,args.arm,args.block,'eval',phase)
            torch.manual_seed(schedule['eval_action_seed']+index)
            sampled=policy.act([prepared],with_reference=True)
            action=sampled['actions'][0]
            nxt=env.step(action)
            score=tracker.score(nxt['transition'],nxt['observation'])
            row=dict(index=index,observation=obs,snapshot=snapshot,prepared=prepared,action=action,
                action_source='model',transition=nxt['transition'],next_observation=nxt['observation'],
                terminal=nxt['terminated'],r1_score=score,duration=score['elapsed_ticks'],value=sampled['values'][0])
            for key,plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                                ('reference_verb','reference_verb'),('reference_target','reference_target')]:
                row[key]=sampled[plural][0]
            rows.append(row)
            append(out/'rollout.jsonl',stored(row))
            obs=nxt['observation']
            if row['terminal']:
                break
    result=dict(full=metrics(rows),first16=metrics(rows,16))
    write(out/'metrics.json',result)
    return result


def training(args):
    import torch
    from agent.m9_policy import save
    from tools.mea_shuffle_train import state, steps, optimizer_gate, training_update
    from .policy import new_policy, load_checked, CONFIG
    started=time.monotonic()
    data=Dataset(args.source)
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    count=16 if args.smoke else 128
    rounds=1 if args.smoke else 3
    config=CONFIG | dict(minibatch=count,critic_minibatch=count)
    registration=dict(protocol=PROTOCOL,config=config,schedule=cell(args.block,args.arm),data=data.summary(),
                      smoke=args.smoke,collection_pid=os.getpid(),code_manifest=json.loads((args.root/'manifest.json').read_text()))
    write(out/'registration.json',registration)
    policy=new_policy(args.model_config,args.arm,args.block,data.normalization_id,count)
    initial=state(policy)
    if initial != ORIGINAL_TENSORS:
        raise ValueError('Initial parameters differ from original pure-R1 warm start')
    write(out/'initial_parameters.json',initial)
    if any('lora_' not in name for name,p in policy.model.named_parameters() if p.requires_grad):
        raise ValueError('Unexpected trainable base parameters')
    opts=dict(actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=config['actor_lr'],eps=1e-5,weight_decay=0),
              critic=torch.optim.AdamW(policy.critic.parameters(),lr=config['critic_lr'],eps=1e-5,weight_decay=0))
    ids={k:[id(p) for g in v.param_groups for p in g['params']] for k,v in opts.items()}
    save(policy,out/'initial_checkpoint',opts['actor'],opts['critic'],0,digest(registration))
    # Same-format pre-update assessment makes prompt adaptation visible separately.
    if not args.smoke:
        pre=evaluate_episode(policy,data,args,'after',out/'pre_after')
        if state(policy)!=initial:
            raise ValueError('Pre-evaluation changed weights')
        write(out/'pre_metrics.json',pre)
    reports=[]
    rewards=Rewards(args.arm in ('B','C','D'))
    schedule=cell(args.block,args.arm)
    with GodotEnv(str(args.godot),str(args.project),map_mode='infinite_freeze',navigation_steps=4) as env:
        seed=schedule['world_seed']
        obs=env.reset(seed)['observation']
        tracker=Reward(obs,'bottleneck',GAMMA)
        for number in range(1,rounds+1):
            rd=out/f'round_{number:02d}'
            rd.mkdir()
            before=state(policy)
            before_steps=steps(opts)
            rows,boot,obs,tracker,seed=collect(policy,data,env,obs,tracker,rewards,seed,args.arm,args.block,
                                            (number-1)*count,count,rd)
            for row in rows:
                append(rd/'rollout.jsonl',stored(row))
            write(rd/'bootstrap.json',boot)
            pre_error=validate_policy_statistics(policy_statistics(policy,[r['prepared'] for r in rows],
                                           [r['verb'] for r in rows],[r['target'] for r in rows]),rows)
            input_hash=digest([evidence(r['prepared']) for r in rows])
            torch.manual_seed(schedule['ppo_seed']+number)
            updated=training_update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**config))
            after_steps=steps(opts)
            applied={k:after_steps[k][0]-(before_steps[k][0] if before_steps[k] else 0) for k in opts}
            optimizer_gate(before_steps,after_steps,applied['actor'],applied['critic'])
            if not 1<=applied['actor']<=2 or applied['critic']!=16 or applied['actor']!=updated['optimizer_steps']:
                raise ValueError('Actual optimizer budget differs')
            after=state(policy)
            columns=bool(torch.count_nonzero(policy.critic[0].weight[:,108:]))
            checks=dict(actor_changed=after['actor']!=before['actor'],critic_changed=after['critic']!=before['critic'],
                frozen_base_unchanged=after['base']==initial['base'],reference_unchanged=after['reference']==initial['reference'],
                input_unchanged=input_hash==digest([evidence(r['prepared']) for r in rows]),
                neural_columns_expected=columns==(args.arm in ('C','D')),
                frozen_base_no_gradient=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad),
                parameters_finite=all(bool(torch.isfinite(p).all()) for m in (policy.model,policy.critic) for p in m.parameters()),
                optimizer_identity_unchanged=ids=={k:[id(p) for g in v.param_groups for p in g['params']] for k,v in opts.items()})
            if not all(checks.values()):
                raise ValueError('Training checks failed: '+str(checks))
            post_stats=policy_statistics(policy,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])
            write(rd/'post_statistics.json',post_stats)
            save(policy,rd/'checkpoint',opts['actor'],opts['critic'],number,digest(registration))
            load_checked(policy,rd/'checkpoint')
            reload_error=validate_policy_statistics(policy_statistics(policy,[r['prepared'] for r in rows],
                                       [r['verb'] for r in rows],[r['target'] for r in rows]),post_stats)
            if state(policy)!=after or steps(opts)!=after_steps:
                raise ValueError('Checkpoint reload changed state')
            report=dict(round=number,decisions=len(rows),before=before,after=after,checks=checks,
                applied_steps=applied,optimizer_steps_before=before_steps,optimizer_steps_after=after_steps,
                metrics=updated,pre_update_error=pre_error,reload_error=reload_error,
                positive=sum(r['reward_record']['r2_raw']==1 for r in rows),
                negative=sum(r['reward_record']['r2_raw']==-1 for r in rows),
                terminals=sum(r['terminal'] for r in rows),r1_sum=math.fsum(r['r1_score']['reward'] for r in rows),
                r2_contribution_sum=math.fsum(r['reward_record']['r2_contribution'] for r in rows))
            write(rd/'report.json',report)
            reports.append(report)
            emit('round_passed',block=args.block,arm=args.arm,round=number,applied=applied)
            del rows,post_stats
            policy.clear_prefix_cache()
            gc.collect()
            torch.cuda.synchronize()
    write(out/'training_result.json',dict(passed=True,model_training_performed=True,smoke=args.smoke,
          rounds=reports,initial=initial,final=state(policy),seconds=time.monotonic()-started,
          peak_cuda_allocated=torch.cuda.max_memory_allocated(),r2_source='simulation_from_game_state',
          hardware_commands_sent=0,biological_effect_verified=False))


def evaluation(args):
    from tools.mea_shuffle_train import state
    from .policy import new_policy,load_checked
    data=Dataset(args.source)
    args.output.mkdir(parents=True,exist_ok=False)
    policy=new_policy(args.model_config,args.arm,args.block,data.normalization_id)
    initial=state(policy)
    if args.arm=='A0':
        from .integrity import check_baseline_identity,baseline_padded_critic_hash
        check_baseline_identity(initial,baseline_padded_critic_hash(policy))
    else:
        if initial!=ORIGINAL_TENSORS:
            raise ValueError('Reconstructed warm start differs')
        load_checked(policy,args.checkpoint)
    evaluated=state(policy)
    results={}
    for phase in ('after','before'):
        results[phase]=evaluate_episode(policy,data,args,phase,args.output/phase)
        if state(policy)!=evaluated:
            raise ValueError('Evaluation must keep weights frozen')
        policy.clear_prefix_cache()
    write(args.output/'evaluation.json',dict(passed=True,block=args.block,arm=args.arm,metrics=results,
          initial=initial,evaluated=evaluated,model_training_performed=False,hardware_commands_sent=0))
    emit('evaluated',block=args.block,arm=args.arm,after=results['after']['full'])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('train','evaluate','verify'))
    for name in ('root','source','project','godot','model-config','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--arm',required=True,choices=('A0','A','B','C','D'))
    p.add_argument('--block',required=True,type=int)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--train-dir',type=Path)
    p.add_argument('--eval-dir',type=Path)
    p.add_argument('--smoke',action='store_true')
    return p


def main():
    args=parser().parse_args()
    cell(args.block,args.arm)
    if args.mode=='train':
        if args.arm=='A0': raise ValueError('Frozen baseline cannot train')
        training(args)
    elif args.mode=='evaluate': evaluation(args)
    else:
        from .verify import verify
        verify(args)


if __name__=='__main__': main()
