"""Train and evaluate one preregistered 28080 offline arm/seed."""
import argparse, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.m9_policy import load, save
from agent.mea_contract import observation_key
from agent.mea_long_ablation import ORIGINAL_TENSORS
from mea_state2_r2.features import NAMES, SCHEMA, digest, transform
from mea_state2_r2.policy import CONFIG, Policy
from mea_state2_r2.protocol import GAMMA
from mea_state2_r2.trajectory import finish
from tools.mea_offline import _value, policy_statistics, validate_policy_statistics
from tools.mea_shuffle_train import state, training_update


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')

class FixedContextPolicy(Policy):
    def __init__(self, config, comparison, context):
        self.comparison = comparison
        self.context = context
        # B masks the four MEA inputs; C passes the recorded inputs. Both use
        # the same warm-started model tensors and optimizer budget.
        super().__init__(config, 'B' if comparison == 'masked' else 'C', digest(context['normalization']))
    def metadata(self):
        m = super().metadata()
        m.pop('stimulus_profile_sha256', None)
        m.update(profile='cpp_28080_low_health_60tick', comparison=self.comparison,
                 state2_source='historical_28080_six_rounds', source_sha=self.context['source_sha256'],
                 alpha=1.0, r2_enabled=False, r2_source='logged_only', beta=0.0,
                 stimulation=False, realtime_training=False)
        return m
    def replay_input(self, obs, window):
        snap = dict(schema=SCHEMA, names=list(NAMES), source='recording',
                    association='historical_no_game_causality', source_pin=self.context['source_sha256'],
                    phase=window['phase'], window_index=window['index'], window_start=window['start_ms'],
                    window_end=window['end_ms'], channel_map_id=self.context['roi_sha'],
                    events_sha256=window['events_sha'], context_id=observation_key(obs),
                    features=window['features'], normalized=window['normalized'],
                    normalization_id=self.normalization_id)
        snap['snapshot_id'] = digest(snap)
        return self.prepare([obs], [snap])[0], snap


def reset_low(env, study, seed):
    reply = env.call('reset', seed=int(seed), profile='m9', map_mode='infinite_freeze',
                     navigation_steps=4, starting_scene=study['starting_scene'],
                     starting_reserves=study['starting_reserves'])
    obs = reply['observation']
    if reply.get('starting_scene') != study['starting_scene']:
        raise ValueError('starting scene was not applied')
    for key, value in study['starting_reserves'].items():
        if abs(float(obs['player'][key]) - float(value)) > 1e-6:
            raise ValueError('starting reserve was not applied: '+key)
    return obs


def collect(policy, env, study, seed, windows, training=True):
    obs = reset_low(env, study, seed)
    tracker = Reward(obs, 'bottleneck', GAMMA)
    rows = []
    for i, window in enumerate(windows):
        prepared, snapshot = policy.replay_input(obs, window)
        torch.manual_seed(int(seed) * 10 + i)
        sample = policy.act([prepared], with_reference=True)
        action = sample['actions'][0]
        nxt = env.step(action)
        score = tracker.score(nxt['transition'], nxt['observation'])
        expected = [0.0] * 4 if policy.comparison == 'masked' else list(snapshot['normalized'])
        encoded = ','.join(f'{n}={v:.3f}' for n, v in zip(NAMES, expected))
        if prepared['wetware_features'] != expected or prepared['critic_input'][-4:] != expected or encoded not in prepared['messages'][1]['content']:
            raise ValueError('MEA features did not follow the registered arm')
        row = dict(index=i, seed=seed, action=action, observation=obs, prepared=prepared,
                   snapshot=snapshot, next_observation=nxt['observation'], transition=nxt['transition'],
                   r1_score=score, duration=score['elapsed_ticks'], reward=score['reward'],
                   r2=0, contribution=0.0, r2_logged=True, terminal=nxt['terminated'],
                   value=sample['values'][0])
        for key, plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                            ('reference_verb','reference_verb'),('reference_target','reference_target')]:
            row[key] = sample[plural][0]
        rows.append(row)
        obs = nxt['observation']
        if row['terminal']:
            if training:
                raise ValueError('training episode ended before its registered budget')
            break
    return rows, obs


def first_success(transition, verb):
    before, after = transition['before'], transition['after']
    if verb == 'drink' and float(after['thirst']) > float(before['thirst']): return True
    if verb == 'eat' and float(after['hunger']) > float(before['hunger']): return True
    return False


def evaluate(policy, env, study, context, world, arm):
    order = list(range(60)) if arm != 'shuffled' else study['eval_order_shuffled'][str(world)]
    obs = reset_low(env, study, world)
    initial_visited = int(obs['visited_count'])
    tracker = Reward(obs, 'bottleneck', GAMMA)
    rows = []
    first_drink = None; first_eat = None
    while int(obs['world_tick']) < int(study['horizon']) and not bool(obs.get('terminated', False)):
        tick = int(obs['world_tick'])
        window = context['holdout_windows'][order[tick]]
        prepared, snapshot = policy.replay_input(obs, window)
        torch.manual_seed(int(world) * 1000 + tick)
        sample = policy.act([prepared], with_reference=False)
        action = sample['actions'][0]
        nxt = env.step(action)
        score = tracker.score(nxt['transition'], nxt['observation'])
        tr = nxt['transition']
        end = tr['after']
        if first_drink is None and first_success(tr, 'drink'): first_drink = int(end['tick'])
        if first_eat is None and first_success(tr, 'eat'): first_eat = int(end['tick'])
        rows.append(dict(tick=tick, window=window['index'], action=action, r1=score['reward'],
                         end_tick=int(end['tick']), transition_sha=digest(tr),
                         new_cells=int(end['visited_count'])-initial_visited,
                         health=float(end['health']), thirst=float(end['thirst']), hunger=float(end['hunger'])))
        obs = nxt['observation']
        if nxt['terminated']: break
    end = rows[-1] if rows else dict(end_tick=int(obs['world_tick']))
    end_tick = int(obs['world_tick'])
    return dict(world=int(world), arm=arm, decisions=len(rows), end_tick=end_tick,
                survived=int(not bool(obs.get('terminated', False)) and end_tick >= int(study['horizon'])),
                survival_at_60=int(not bool(obs.get('terminated', False)) and end_tick >= int(study['horizon'])),
                survival_duration=end_tick, r1_sum=sum(float(x['r1']) for x in rows),
                new_cells=int(obs['visited_count'])-initial_visited,
                first_success_drink=first_drink, first_success_eat=first_eat,
                final_health=float(obs['player']['health']), final_thirst=float(obs['player']['thirst']),
                final_hunger=float(obs['player']['hunger']), rows=rows)


def run(args):
    root = Path(args.root); study = read(root/'design.json'); context = read(root/'context.json')
    if digest(context) != study['context_sha']: raise ValueError('context digest mismatch')
    if args.arm not in study['arms'] or int(args.seed) not in study['training_seeds']:
        raise ValueError('unregistered arm or seed')
    out = root/'models'/str(args.seed)/args.arm
    out.mkdir(parents=True, exist_ok=False)
    config = CONFIG | dict(minibatch=study['batch'], critic_minibatch=study['batch'])
    paths = read(args.model_config)
    torch.manual_seed(6100000)
    policy = FixedContextPolicy(SimpleNamespace(**config, **paths, rng_seed=6100000), args.arm, context)
    initial = state(policy)
    if initial != ORIGINAL_TENSORS: raise ValueError('unexpected warm-start tensor identity')
    actor = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=config['actor_lr'], eps=1e-5, weight_decay=0)
    critic = torch.optim.AdamW(policy.critic.parameters(), lr=config['critic_lr'], eps=1e-5, weight_decay=0)
    order = list(range(48)) if args.arm != 'shuffled' else study['train_order_shuffled'][str(args.seed)]
    train_windows = [context['train_windows'][i] for i in order]
    registration = dict(study_digest=digest(study), context_digest=digest(context), initial=initial,
                        config=config, training_seed=int(args.seed), arm=args.arm,
                        source_sha=context['source_sha256'], train_order=[w['index'] for w in train_windows],
                        holdout_worlds=study['holdout_worlds'], r2_mode='logged_only', alpha=1.0, beta=0.0,
                        starting_scene=study['starting_scene'], starting_reserves=study['starting_reserves'])
    write(out/'registration.json', registration)
    rounds=[]
    with GodotEnv(str(args.godot), str(args.project), map_mode='infinite_freeze', navigation_steps=4) as env:
        for run_index, world_seed in enumerate(study['episodes'][str(args.seed)], start=1):
            batch = train_windows[(run_index-1)*study['batch']:run_index*study['batch']]
            rows, obs = collect(policy, env, study, world_seed, batch, True)
            boot = 0.0 if rows[-1]['terminal'] else _value(policy, policy.replay_input(obs, batch[-1])[0])
            finish(rows, boot, GAMMA, config['gae_lambda'])
            with (out/f'train-{run_index}.jsonl').open('w',encoding='utf8') as f:
                for row in rows:
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            def stats(): return policy_statistics(policy, [r['prepared'] for r in rows], [r['verb'] for r in rows], [r['target'] for r in rows])
            validate_policy_statistics(stats(), rows)
            torch.manual_seed(int(args.seed)+210+run_index)
            update = training_update(policy, actor, critic, rows, SimpleNamespace(**config))
            if update.get('optimizer_steps') != 2: raise ValueError('unexpected PPO budget')
            rounds.append(dict(round=run_index, world_seed=world_seed, update=update, decisions=len(rows), bootstrap=boot,
                               r2_logged=sum(1 for r in rows if r['r2_logged']), r2_contribution=0.0))
        after = state(policy)
        if any((after[k] != initial[k]) != (k in ('actor','critic')) for k in initial): raise ValueError('unexpected frozen tensor change')
        save(policy, out/'checkpoint', actor, critic, len(rounds), digest(registration))
        load(policy, out/'checkpoint')
        if state(policy) != after: raise ValueError('checkpoint round-trip mismatch')
        evaluation=[]
        for world in study['holdout_worlds']:
            ev=evaluate(policy,env,study,context,world,args.arm)
            evaluation.append(ev); write(out/f'eval-{world}.json',ev)
        if state(policy) != after: raise ValueError('evaluation changed policy weights')
    result=dict(arm=args.arm,training_seed=int(args.seed),initial=initial,after=after,rounds=rounds,
                train_decisions=48,holdout_windows=60,eval=evaluation,
                checkpoint_sha=digest(read(out/'checkpoint/manifest.json')),
                metadata=policy.metadata())
    write(out/'update.json',dict(initial=initial,after=after,rounds=rounds,actor_steps=sorted({int(v['step']) for v in actor.state.values()}),critic_steps=sorted({int(v['step']) for v in critic.state.values()})))
    write(out/'result.json',result)
    print(json.dumps({'arm':args.arm,'seed':args.seed,'done':True,'new_cells':[v['new_cells'] for v in evaluation]},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('root','project','godot','model-config'): p.add_argument('--'+field,required=True)
    p.add_argument('--arm',choices=['ordered','masked','shuffled'],required=True)
    p.add_argument('--seed',type=int,required=True)
    run(p.parse_args())
