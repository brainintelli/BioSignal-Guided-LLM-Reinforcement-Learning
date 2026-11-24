"""Equal-budget PPO with masked, zero-output, regular or irregular recorded MEA inputs."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.m9_policy import load, save
from agent.mea_contract import observation_key
from agent.mea_long_ablation import ORIGINAL_TENSORS
from livepair.predictability import outcome
from mea_state2_r2.features import NAMES, SCHEMA, digest
from mea_state2_r2.policy import CONFIG, Policy
from mea_state2_r2.protocol import GAMMA
from mea_state2_r2.train import metrics, stored
from mea_state2_r2.trajectory import finish
from tools.mea_offline import _value, policy_statistics, validate_policy_statistics
from tools.mea_shuffle_train import state, training_update


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path,value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False)


class FixedContextPolicy(Policy):
    def __init__(self, config, comparison, context):
        self.comparison=comparison
        self.context=context
        super().__init__(config,'A' if comparison=='masked' else 'C',digest(context['normalization']))

    def metadata(self):
        m=super().metadata()
        m.pop('stimulus_profile_sha256',None)
        return m | dict(profile=self.context.get('profile','22021_fixed_spike_context_ppo'),comparison=self.comparison,
                        state2_source=self.context.get('state2_source','22021_provisional_spike_window'),source_sha=self.context['source_sha256'],
                        r2_enabled=False,r2_source='disabled',beta=0.,
                        stimulation=False,realtime_training=False)

    def replay_input(self,obs,window=None):
        c=self.context if window is None else self.context | window
        snap=dict(schema=SCHEMA,names=list(NAMES),source='recording',association='historical_no_game_causality',
                  source_pin=c['source_sha256'],phase='after',window_index=c.get('index',0),window_start=c['frame_start'],
                  window_end=c['frame_end'],channel_map_id=c['roi_sha'],events_sha256=c['events_sha'],
                  context_id=observation_key(obs),features=c['features'],normalized=c['normalized'],
                  normalization_id=self.normalization_id)
        snap['snapshot_id']=digest(snap)
        return self.prepare([obs],[snap])[0],snap


def collect(policy,env,study,seed,length,training,windows=None):
    initial=env.reset(seed)
    obs=initial['observation']
    if initial.get('starting_scene')!=study['starting_scene'] or any(obs['player'][k]!=v for k,v in study['starting_reserves'].items()):
        raise ValueError('Wrong starting scene')
    tracker=Reward(obs,'bottleneck',GAMMA)
    rows=[]
    for i in range(length):
        prepared,snapshot=policy.replay_input(obs,None if windows is None else windows[i])
        torch.manual_seed(seed*10+i)
        sample=policy.act([prepared],with_reference=True)
        action=sample['actions'][0]
        nxt=env.step(action)
        transition=nxt['transition']
        score=tracker.score(transition,nxt['observation'])
        ticks=score['elapsed_ticks']
        correct=outcome(transition)['correct']
        eligible=(i+1)%study['feedback_stride']==0 and not nxt['terminated']
        sign=(1 if correct else -1) if eligible else 0
        weight=0.
        expected=[0.]*4 if policy.comparison=='masked' else snapshot['normalized']
        encoded=','.join(f'{n}={v:.3f}' for n,v in zip(NAMES,expected))
        if prepared['wetware_features']!=expected or prepared['critic_input'][-4:]!=expected or encoded not in prepared['messages'][1]['content']:
            raise ValueError('Actor or critic received unintended MEA features')
        row=dict(index=i,seed=seed,action=action,observation=obs,prepared=prepared,snapshot=snapshot,
                 next_observation=nxt['observation'],transition=transition,r1_score=score,duration=ticks,
                 reward=score['reward']+weight,r2=sign,contribution=weight,terminal=nxt['terminated'],value=sample['values'][0])
        for k,plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                         ('reference_verb','reference_verb'),('reference_target','reference_target')]:
            row[k]=sample[plural][0]
        rows.append(row)
        obs=nxt['observation']
        if row['terminal'] and training and i+1<length:
            raise ValueError('Training episode ended before its registered budget')
        if row['terminal']:
            break
    return rows,obs


def steps(optimizer):
    return sorted({int(p['step']) for p in optimizer.state.values()})


def train(args):
    root=args.root
    study=read(root/'design.json')
    context=read(root/(args.arm+'.json'))
    if digest(context)!=study['contexts'][args.arm] or args.seed not in study['training_seeds'] or args.arm not in study['arms']:
        raise ValueError('Study binding mismatch')
    config=CONFIG | dict(minibatch=study['batch'],critic_minibatch=study['batch'])
    out=root/'models'/str(args.seed)/args.arm
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(6100000)
    policy=FixedContextPolicy(SimpleNamespace(**config,**read(args.model_config),rng_seed=6100000),args.arm,context)
    initial=state(policy)
    if initial!=ORIGINAL_TENSORS:
        raise ValueError('Initial checkpoint mismatch')
    actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=config['actor_lr'],eps=1e-5,weight_decay=0)
    critic=torch.optim.AdamW(policy.critic.parameters(),lr=config['critic_lr'],eps=1e-5,weight_decay=0)
    registration=dict(study_digest=digest(study),initial=initial,config=config,training_seed=args.seed,arm=args.arm,
                      source_sha=context['source_sha256'])
    series=context.get('series')
    if series is None:
        registration['window']=[context['frame_start'],context['frame_end']]
    else:
        if len(series['train'])!=study['batch']*study['updates'] or len(series['holdout'])!=study['eval_decisions'] or len(series['bootstrap'])!=study['updates']:
            raise ValueError('Recorded window count differs from the registered budgets')
        registration['windows_digest']=digest(series)
    write(out/'registration.json',registration)
    rounds=[]
    with GodotEnv(str(args.godot),str(args.project),map_mode='infinite_freeze',navigation_steps=4) as env:
        for run,seed in enumerate(study['episodes'][str(args.seed)],start=1):
            batch_windows=None if series is None else series['train'][(run-1)*study['batch']:run*study['batch']]
            rows,obs=collect(policy,env,study,seed,study['batch'],True,batch_windows)
            # Each reset world contributes a fixed horizon; bootstrap a nonterminal last state.
            bootstrap_window=None if series is None else series['bootstrap'][run-1]
            boot=0. if rows[-1]['terminal'] else _value(policy,policy.replay_input(obs,bootstrap_window)[0])
            finish(rows,boot,GAMMA,config['gae_lambda'])
            with (out/f'train-{run}.jsonl').open('x',encoding='utf8') as stream:
                for row in rows:
                    stream.write(json.dumps(stored(row),ensure_ascii=False,allow_nan=False)+'\n')
            def stats():
                return policy_statistics(policy,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])
            validate_policy_statistics(stats(),rows)
            torch.manual_seed(args.seed+210+run)
            update=training_update(policy,actor,critic,rows,SimpleNamespace(**config))
            if update['optimizer_steps']!=2 or steps(actor)!=[2*run] or steps(critic)!=[16*run]:
                raise ValueError('Different effective PPO budgets')
            rounds.append(dict(world=seed,update=update,decisions=len(rows),bootstrap=boot,r2=[r['r2'] for r in rows],
                               contribution=sum(r['contribution'] for r in rows)))
            print('round complete',run,flush=True)
        after=state(policy)
        if any((after[k]!=initial[k])!=(k in ('actor','critic')) for k in initial):
            raise ValueError('Unexpected trainable/frozen weights')
        post=stats()
        save(policy,out/'checkpoint',actor,critic,len(rounds),digest(registration))
        load(policy,out/'checkpoint')
        validate_policy_statistics(stats(),post)
        if state(policy)!=after:
            raise ValueError('Checkpoint reload mismatch')
        write(out/'update.json',dict(initial=initial,after=after,rounds=rounds,actor_steps=steps(actor),critic_steps=steps(critic)))
        evaluation=[]
        with (out/'holdout.jsonl').open('x',encoding='utf8') as stream:
            for world in study['holdout_seeds']:
                batch,_=collect(policy,env,study,world,study['eval_decisions'],False,None if series is None else series['holdout'])
                m=metrics(batch,horizon=study['eval_decisions'])
                m.update(seed=world,new_cells=batch[-1]['transition']['after']['visited_count']-batch[0]['transition']['before']['visited_count'])
                evaluation.append(m)
                audit=[dict(context=r['snapshot']['context_id'],action=r['action'],after=r['transition']['after'],
                            r1=r['r1_score']['reward'],r2=r['r2']) for r in batch]
                if series is not None:
                    for event,row in zip(audit,batch):
                        event['consumed_features']=row['prepared']['wetware_features']
                        event['wetware_window']=row['snapshot']['window_index']
                        event['wetware_events_sha']=row['snapshot']['events_sha256']
                stream.write(json.dumps(dict(world=world,rows=audit),ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
        if state(policy)!=after:
            raise ValueError('Evaluation mutated weights')
        write(out/'result.json',dict(evaluation=evaluation,checkpoint_digest=digest(read(out/'checkpoint/manifest.json'))))
        print('complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('root','project','godot','model-config'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--arm',choices=['baseline'],required=True)
    train(parser.parse_args())
