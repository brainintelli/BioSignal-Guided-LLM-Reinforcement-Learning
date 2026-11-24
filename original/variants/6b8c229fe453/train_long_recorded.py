"""Repeat the registered train/holdout windows across continuous game trajectories."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from livepair.protocol import check_window, snapshot
from mea_online.m9_runtime import (DEFAULTS, GAME_ONLY, GAME_R2, NATIVE_GAME_STUDIES,
                                   STATE2_ONLY, M9Runtime, OFFLINE_PULSE, _materialize, digest)
from mea_state2_r2.protocol import BETA, GAMMA
from train_recorded_pulses import metrics, windows, write


def read(path):
    return json.loads(path.read_bytes())


def progress(folder, updates, games):
    tmp = folder / 'progress.tmp'
    tmp.write_text(json.dumps(dict(updates=updates, games=games)))
    tmp.replace(folder / 'progress.json')


def prepare(runtime, config, obs, donor, key):
    from agent.m9_contract import encode
    if config.get('experiment') in NATIVE_GAME_STUDIES:
        if donor is not None:
            raise ValueError('Physical donor in the pure R1 arm')
        return runtime.prepare_game(obs)
    game_id = digest(encode(obs, DEFAULTS['state_variant']))
    return runtime.prepare(obs, snapshot(donor, config, game_id, key))


def decisions(runtime, config, order, prime, seed, horizon, train):
    import torch
    obs = runtime.reset(seed)
    prior = prime
    episodes = 0
    phase = 'train' if train else 'eval'
    pure = config.get('experiment') == GAME_ONLY
    native = config.get('experiment') in NATIVE_GAME_STUDIES
    for i in range(horizon):
        source, donor, window_type, window = (None, None, None, None) if pure else order[i % len(order)]
        prepared = prepare(runtime, config, obs, prior, f'{phase}-{seed}-{i}')
        if native:
            from agent.m9_contract import encode
            native_messages = encode(obs, DEFAULTS['state_variant'])['messages']
            if ('snapshot' in prepared or len(prepared['critic_input']) != 108 or
                    prepared['messages'] != native_messages):
                raise ValueError('Non-native input in pure R1 trajectory')
        obs_hash = digest(obs)
        torch.manual_seed(seed * 10 + i)
        action = runtime.act(prepared)
        if not train:
            with torch.no_grad():
                materialized = _materialize(runtime.core.policy, prepared)
                probs = runtime.core.policy.verb_distribution([materialized]).probs[0].tolist()
        step = runtime.step(action['action'])
        ticks, terminal = step['duration'], step['terminal']
        sign = None if not config['r2_enabled'] or terminal else (1 if window_type == 'pre' else -1)
        addition = (0.0 if sign is None else
                    BETA * sign * math.expm1(ticks * math.log(GAMMA)) / math.expm1(math.log(GAMMA)))
        row = dict(**action, prepared=prepared, terminal=terminal, boundary=terminal, duration=ticks,
                   reward=step['reward'] + addition, r1=step['reward'], r2=sign, r2_contribution=addition,
                   transition=step['transition'], context_donor=None if native else prior['id'],
                   next_donor=None if native else window['id'],
                   r2_trial=source if config['r2_enabled'] else None, next_state2_trial=donor, recorded_phase=window_type)
        obs, prior = step['observation'], window
        trace = {k: v for k, v in row.items()
                 if k not in ('prepared', 'reference_verb', 'reference_target', 'transition')}
        trace.update(decision=i+1, episode=episodes, obs_hash=obs_hash)
        if native:
            trace.update(actor_native_messages=True,critic_input_size=len(prepared['critic_input']))
        elif config['experiment']==STATE2_ONLY:
            trace.update(critic_input_size=len(prepared['critic_input']),state2=prepared['snapshot']['normalized'])
        if not train:
            trace['verb_probabilities'] = probs
            for when in ('before', 'after'):
                trace[when] = {k: row['transition'][when][k]
                               for k in ('visited_count', 'health', 'hunger', 'thirst')}
        yield row, trace, obs, prior
        if terminal:
            if not train:
                return
            if i+1 < horizon:
                episodes += 1
                obs, prior = runtime.reset(seed + 10000 * episodes), prime


def run(args):
    import torch
    budget = read(args.budget)
    batch = budget['rollout_batch']
    pure = args.arm == 'pure'
    if pure:
        binding = budget['replay_sha256']
        prime, train, test = None, [], []
        source_cfg=dict(source='game',experiment=GAME_ONLY,r2_enabled=False,normalization_id='unused',
                        probe_reference='rtx4090-20261116')
        physical_files, replay_digest = 0, None
    else:
        data = read(args.data)
        if hashlib.sha256(args.data.read_bytes()).hexdigest() != budget['replay_sha256']:
            raise ValueError('Unregistered replay file')
        study = data['study']
        binding = digest(study)
        prime = read(args.prime)['prime']
        check_window(prime, binding)
        train, test = (windows(data, args.arm, training) for training in (True, False))
        registered = budget.get('registered_training_seeds',study['training_seeds'])
        if (len(train) != batch or len(test) != 6 or set(r[0] for r in train) & set(r[0] for r in test)
                or not set(budget['training_seeds']) <= set(registered)
                or not set(study['game_eval_seeds']).issubset(set(budget['game_eval_seeds']))):
            raise ValueError('Unmatched budget or physical split')
        source_cfg=dict(source='recorded',experiment=STATE2_ONLY,
                        r2_enabled=False,
                        normalization=data['normalization'], normalization_id=digest(data['normalization']))
        if args.arm=='r2only':
            prime=None
            train,test=([(i,None,phase,None) for i,_,phase,_ in order] for order in (train,test))
            source_cfg=dict(source='game',experiment=GAME_R2,r2_enabled=True,normalization_id='unused',
                            probe_reference='rtx4090-20261116')
        physical_files, replay_digest = len(data['trials']),digest(data)
    if budget['train_steps'] % batch or len(set(budget['training_seeds']))!=len(budget['training_seeds']):
        raise ValueError('Invalid training budget')
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in budget['training_seeds']:
        out = args.out / str(seed)
        out.mkdir()
        cfg = dict(**source_cfg, stimulation=False,
                   arm=('stateonly' if args.arm in ('ordered','shuffled') else args.arm), binding_sha256=binding, devices=list(range(8)), batch_size=batch,
                   seed=seed, training=dict(minibatch=batch, critic_minibatch=batch), **read(args.settings))
        reg = dict(config=cfg, budget=budget, physical_files=physical_files, replay_digest=replay_digest,
                   eval_seeds=budget['game_eval_seeds'], train_pairs=[r[:3] for r in train],
                   test_pairs=[r[:3] for r in test])
        write(out/'registration.json', reg)
        torch.manual_seed(seed)
        rt = M9Runtime(cfg, out)
        try:
            initial = rt.state()
            updates, traces, rows = [], [], []
            for row, trace, obs, prior in decisions(rt, cfg, train, prime, seed, budget['train_steps'], True):
                rows.append(row)
                traces.append(trace)
                if len(rows) == batch:
                    uid = len(updates) + 1
                    boot = 0.0 if row['terminal'] else rt.value(prepare(rt, cfg, obs, prior, f'bootstrap-{seed}-{uid}'))
                    update = rt.update(rows, boot, uid)
                    if not (update['world_size'] == 8 and update['unique_decisions'] == batch and
                            update['actor_changed'] and update['critic_changed']):
                        raise ValueError('Incomplete eight-rank update')
                    updates.append(update)
                    progress(out, uid, 0)
                    rows = []
            write(out/'training-trace.json', traces)
            write(out/'updates.json', updates)
            saved = rt.checkpoint(out/'checkpoint', digest(reg), len(updates))
            after = rt.state()
            evals, eval_traces = [], {}
            for gs in budget['game_eval_seeds']:
                ev, detail = [], []
                for row, trace, _, _ in decisions(rt, cfg, test, prime, gs, budget['eval_horizon'], False):
                    ev.append(row)
                    detail.append(trace)
                evals.append(dict(seed=gs, **metrics(ev)))
                eval_traces[gs] = detail
                progress(out, len(updates), len(evals))
            if rt.state() != after:
                raise ValueError('Evaluation changed model state')
            write(out/'eval-trace.json', eval_traces)
            result = dict(training_seed=seed, arm=args.arm, initial=initial, after=after, update=updates[-1],
                          native_critic_sha256=rt.core.policy.native_critic_sha256,
                          checkpoint=saved, training_steps=len(traces), update_count=len(updates), eval=evals)
            write(out/'result.json', result)
            results.append(result)
        finally:
            rt.close()
    write(args.out/'result.json', dict(arm=args.arm, study=binding, replay_digest=replay_digest,
                                      budget=budget, seeds=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('data', 'prime', 'settings', 'budget', 'out'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--arm', choices=('ordered','shuffled','pure','stateonly','r2only'), required=True)
    run(p.parse_args())
