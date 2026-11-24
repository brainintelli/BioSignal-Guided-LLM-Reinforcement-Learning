"""Offline game PPO from the current crossover's C++ observations. No device I/O."""
import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.m9_policy import save,load
from agent.mea_contract import observation_key
from agent.mea_long_ablation import ORIGINAL_TENSORS
from livepair.predictability import outcome
from mea_state2_r2.features import NAMES,SCHEMA,digest,transform
from mea_state2_r2.policy import CONFIG,Policy
from mea_state2_r2.protocol import BETA,GAMMA
from mea_state2_r2.train import metrics,stored
from mea_state2_r2.trajectory import finish
from tools.mea_offline import _value,policy_statistics,validate_policy_statistics
from tools.mea_shuffle_train import state,training_update


class ReplayPolicy(Policy):
    def __init__(self,config,comparison,normalization_id):
        self.comparison=comparison
        super().__init__(config,'B' if comparison=='masked' else 'C',normalization_id)

    def metadata(self):
        m=super().metadata()
        m.pop('stimulus_profile_sha256',None)
        m.update(profile='cpp_28080_offline_game_ppo',comparison=self.comparison,
                 r2_enabled=self.comparison!='r1',r2_source='simulated_game_correctness' if self.comparison!='r1' else 'disabled',
                 stimulation=False,realtime_training=False)
        return m


def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False,indent=2)


def run(args):
    data=json.loads(args.data.read_text());pin=digest(data); norm=data['normalization']; norm_id=digest(norm)
    spec=json.loads(args.study.read_text(encoding='utf8')) if args.study else None
    train_seed=args.train_seed
    if spec and (train_seed not in spec['training_seeds'] or args.arm not in spec['arms'] or digest(data)!=spec['replay_digest']):
        raise ValueError('Replay input or training assignment differs from registration')
    arm=args.arm;out=args.out;out.mkdir(exist_ok=False,parents=True)
    train=[r for r in data['trials'] if r['block']<=4]; test=[r for r in data['trials'] if r['block']>4]
    if len(train)!=12 or len(test)!=6 or set(r['index'] for r in train)&set(r['index'] for r in test):
        raise ValueError('Require all prespecified train and holdout trials')
    if arm=='shuffled':
        rng=random.Random(740019)
        for start in range(0,len(train),3):
            perm=list(range(3))
            while any(i==x for i,x in enumerate(perm)):rng.shuffle(perm)
            train[start:start+3]=[train[start+i] for i in perm]
    config=CONFIG | dict(minibatch=12,critic_minibatch=12)
    paths=json.loads(args.model_config.read_text())
    torch.manual_seed(6100000)
    policy=ReplayPolicy(SimpleNamespace(**config,**paths,rng_seed=6100000),arm,norm_id)
    initial=state(policy)
    if initial!=ORIGINAL_TENSORS:raise ValueError('Unexpected initial warm start')
    opts=dict(actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=config['actor_lr'],eps=1e-5,weight_decay=0),
              critic=torch.optim.AdamW(policy.critic.parameters(),lr=config['critic_lr'],eps=1e-5,weight_decay=0))
    registration=dict(arm=arm,source_sha=pin,initial=initial,normalization_id=norm_id,train_order=[r['index'] for r in train],
        test_order=[r['index'] for r in test],source_clock='C++ host milliseconds',r2='game-simulated',real_game_contingent_outputs=0,
        training_seed=train_seed,eval_seeds=spec['holdout_seeds'] if spec else list(range(840001,840009)),config=config)
    if spec:
        registration.update(starting_scene=spec['starting_scene'],study_digest=digest(spec))
    write(out/'registration.json',registration)

    def prepare(obs,w):
        snap=dict(schema=SCHEMA,names=list(NAMES),source='recording',association='historical_no_game_causality',
            source_pin=pin,phase=w['phase'],window_index=w['index'],window_start=w['start_ms'],window_end=w['start_ms']+1000,
            channel_map_id=data['roi_sha'],events_sha256=w['events_sha'],context_id=observation_key(obs),
            features=w['features'],normalized=list(transform(w['features'],norm)),normalization_id=norm_id)
        snap['snapshot_id']=digest(snap)
        return policy.prepare([obs],[snap])[0],snap

    def episode(env,seed,order,train_phase):
        def reset(world_seed):
            reply=env.reset(world_seed)
            obs=reply['observation']
            if spec:
                reserves={key:obs['player'][key] for key in ('health','thirst','hunger')}
                if reply.get('starting_scene')!=spec['starting_scene'] or reserves!=spec['starting_reserves']:
                    raise ValueError('Game reset did not apply registered starting scene')
            return obs
        obs=reset(seed);tracker=Reward(obs,'bottleneck',GAMMA)
        rows=[];resets=0
        for i,w in enumerate(order):
            prep,snap=prepare(obs,w)
            torch.manual_seed(seed*10+i)
            sampled=policy.act([prep],with_reference=True)
            action=sampled['actions'][0];nxt=env.step(action)
            native=tracker.score(nxt['transition'],nxt['observation'])
            n=native['elapsed_ticks'];terminal=nxt['terminated']
            correct=outcome(nxt['transition'])['correct']
            r2=(1 if correct else -1) if i%3==2 and not terminal else 0
            add=BETA*r2*math.expm1(n*math.log(GAMMA))/math.expm1(math.log(GAMMA)) if arm!='r1' else 0.
            row=dict(index=i,observation=obs,action=action,snapshot=snap,prepared=prep,transition=nxt['transition'],
                next_observation=nxt['observation'],terminal=terminal,duration=n,value=sampled['values'][0],
                reward=native['reward']+add,r1_score=native,reward_record=dict(correct=correct,simulated_r2=r2,
                r2_contribution=add,actual_feedback_to_game=False))
            for k,m in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                        ('reference_verb','reference_verb'),('reference_target','reference_target')]:row[k]=sampled[m][0]
            rows.append(row);obs=nxt['observation']
            if terminal:
                if not train_phase:break
                resets+=1;obs=reset(seed+resets*10000);tracker=Reward(obs,'bottleneck',GAMMA)
        return rows,obs

    with GodotEnv(str(args.godot),str(args.project),map_mode='infinite_freeze',navigation_steps=4) as env:
        rows,obs=episode(env,train_seed,train,True)
        bootstrap=0. if rows[-1]['terminal'] else _value(policy,prepare(obs,data['bootstrap'])[0])
        finish(rows,bootstrap,GAMMA,.999)
        with (out/'train.jsonl').open('x',encoding='utf8') as f:
            for row in rows:f.write(json.dumps(stored(row),ensure_ascii=False,allow_nan=False)+'\n')
        def stats():return policy_statistics(policy,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])
        validate_policy_statistics(stats(),rows)
        torch.manual_seed(train_seed+210)
        updates=training_update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**config))
        after=state(policy)
        if after['actor']==initial['actor'] or after['critic']==initial['critic'] or any(after[k]!=initial[k] for k in ('base','reference')):
            raise ValueError('Incorrect optimizer or frozen parameter changes')
        post_stats=stats()
        save(policy,out/'checkpoint',opts['actor'],opts['critic'],1,digest(registration))
        load(policy,out/'checkpoint')
        validate_policy_statistics(stats(),post_stats)
        if state(policy)!=after:raise ValueError('Checkpoint reload mismatch')
        train_report=dict(initial=initial,after=after,optimizer=updates,r1=sum(r['r1_score']['reward'] for r in rows),
            simulated_r2=[r['reward_record']['simulated_r2'] for r in rows],r2_contribution=sum(r['reward_record']['r2_contribution'] for r in rows))
        write(out/'update.json',train_report)
        evaluation=[]
        for seed in registration['eval_seeds']:
            vals,_=episode(env,seed,test,False)
            m=metrics(vals,horizon=len(test))
            m.update(seed=seed,new_cells=vals[-1]['transition']['after']['visited_count']-vals[0]['transition']['before']['visited_count'])
            write(out/('eval-%d.json'%seed),[dict(index=v['snapshot']['window_index'],action=v['action'],
                r1=v['r1_score']['reward'],simulated_r2=v['reward_record']['simulated_r2'],
                observation_key=v['snapshot']['context_id'],after=v['transition']['after']) for v in vals])
            evaluation.append(m)
        if state(policy)!=after:raise ValueError('Evaluation changed weights')
        write(out/'result.json',dict(arm=arm,train_decisions=len(rows),holdout_windows=len(test),eval=evaluation,
             checkpoint_sha=digest(json.loads((out/'checkpoint/manifest.json').read_text())),**train_report))
        print(json.dumps(dict(arm=arm,done=True,eval=evaluation),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('data','out','model-config','godot','project'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--arm',choices=['ordered','r1','shuffled','masked'],required=True)
    p.add_argument('--study',type=Path)
    p.add_argument('--train-seed',type=int,default=740001)
    run(p.parse_args())
