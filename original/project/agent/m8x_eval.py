"""Open3 frozen evaluation with four physical slots and independent episode RNG.

Refill changes utilization, never actions, rewards, temperatures, or world rules.
Numerical batch/refill invariance must be checked on the real GPU before launch.
"""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path

from agent.m8_eval import DECODING, episode_seed, summarize_episodes


class SlotSchedule:
    def __init__(self, identities, size=4):
        pairs = list(identities)
        if (size != 4 or not pairs or len(set(pairs)) != len(pairs)
                or any(type(s) is not int or type(r) is not int or r < 0 for s, r in pairs)):
            raise ValueError('Invalid fixed-slot evaluation matrix')
        self.pending = deque(pairs)
        self.active = {slot: self.pending.popleft() for slot in range(min(size, len(pairs)))}

    def complete(self, slot):
        if slot not in self.active:
            raise ValueError('An inactive episode cannot complete twice')
        del self.active[slot]
        if self.pending:
            self.active[slot] = self.pending.popleft()
            return self.active[slot]
        return None


def target_encoder(contract):
    if contract == 'open_world_hierarchical_v2':
        from agent.m6_contract import target_messages
    elif contract == 'open_world_exploration_v1':
        from agent.m8x_contract import target_messages
    else:
        raise ValueError('Unknown balanced evaluation contract')
    return target_messages


def evaluate(policy, c, seeds, destination, arm='candidate'):
    import torch
    from agent.rl_contract import transition_at_horizon
    from agent.m8x_identity import source_identity, sha
    legacy = arm == 'original_v2'
    if arm not in ('original_v2', 'balanced_parent', 'candidate'):
        raise ValueError('Unknown evaluation arm')
    if legacy:
        from agent.rl_env import GodotEnv
        from agent.m6_contract import SurvivalReward
    else:
        from agent.m8x_env import GodotEnv
        from agent.m8x_contract import SurvivalReward
    profile = 'open1' if legacy else 'open3'
    world_version = 2 if legacy else 5
    environment_version = None if legacy else 'm8_exploration_v1'
    balance_sha = None if legacy else sha(Path(c.project) / 'configs/m8_exploration_v1.json')
    metadata = policy.metadata()
    contract = metadata['contract']
    expected_contract = 'open_world_hierarchical_v2' if legacy else 'open_world_exploration_v1'
    if (c.profile != profile or contract != expected_contract or c.eval_batch_size != 4
            or c.temperature != 1 or c.horizon != 1800 or list(c.replicates) != [0]):
        raise ValueError('Evaluation world, contract, temperature, shape or matrix changed')
    actor_digest = policy.digest()
    target_messages = target_encoder(contract)
    destination = Path(destination)
    trace = destination.with_suffix('.jsonl')
    if destination.exists() or trace.exists():
        raise FileExistsError('Preserve previous assessment evidence')
    schedule = SlotSchedule([(seed, replicate) for replicate in c.replicates for seed in seeds])
    reports, envs, states, trackers, generators, episodes = [], {}, {}, {}, {}, {}

    def begin(slot):
        seed, replicate = schedule.active[slot]
        state = envs[slot].reset(seed, profile)
        observation = state['observation']
        if (observation.get('world_version') != world_version or observation.get('profile') != profile
                or observation.get('environment_version') != environment_version):
            raise ValueError('Actual Godot world identity differs from the registered arm')
        if not legacy and (metadata.get('balance_sha256') != balance_sha
                           or observation.get('balance_sha256') != balance_sha):
            raise ValueError('Policy rule digest differs from actual project rules')
        rng_seed = episode_seed(c.rng_prefix, profile, seed, replicate)
        states[slot], trackers[slot] = state, SurvivalReward(observation)
        generators[slot] = torch.Generator(device=policy.device_name).manual_seed(rng_seed)
        episodes[slot] = dict(seed=seed, replicate=replicate, rng_seed=rng_seed, decisions=0,
            source_counts={'model': 0, 'fallback': 0}, actions=Counter())

    try:
        for slot in sorted(schedule.active):
            envs[slot] = GodotEnv(c.godot, c.project)
            begin(slot)
        with torch.no_grad(), trace.open('x', encoding='utf-8') as stream:
            while schedule.active:
                active = sorted(schedule.active)
                prepared = policy.prepare([states[i]['observation'] for i in active])
                vd = policy.verb_distribution(prepared)
                before = [hashlib.sha256(generators[i].get_state().cpu().numpy().tobytes()).hexdigest() for i in active]
                verbs = [int(torch.multinomial(vd.probs[j], 1, generator=generators[i]).item()) for j, i in enumerate(active)]
                td = policy.target_distribution(prepared, verbs)
                targets = [int(torch.multinomial(td.probs[j], 1, generator=generators[i]).item()) for j, i in enumerate(active)]
                finished = []
                for j, i in enumerate(active):
                    action = prepared[j]['branches'][verbs[j]][targets[j]]
                    nxt = envs[i].step(action)
                    scored = transition_at_horizon(nxt['transition'], c.horizon)
                    if legacy:
                        reward = trackers[i].score(scored)
                    else:
                        reward, scored = score_evaluation_transition(trackers[i], nxt['transition'], nxt['observation'], c.horizon)
                    final = scored['after']
                    terminal = bool(final['dead'])
                    truncated = final['tick'] >= c.horizon and not terminal
                    after = hashlib.sha256(generators[i].get_state().cpu().numpy().tobytes()).hexdigest()
                    record = dict(arm=arm, seed=episodes[i]['seed'], replicate=episodes[i]['replicate'],
                        rng_seed=episodes[i]['rng_seed'], observation=states[i]['observation'],
                        model_input=prepared[j]['messages'],
                        target_input=target_messages(prepared[j], verbs[j]) if len(prepared[j]['branches'][verbs[j]]) > 1 else None,
                        verb_index=verbs[j], target_index=targets[j], verb_probabilities=vd.probs[j].tolist(),
                        target_probabilities=td.probs[j].tolist(), rng_before_sha256=before[j], rng_after_sha256=after,
                        action=action, transition=nxt['transition'], next_observation=nxt['observation'],
                        reward_scope='scored_horizon' if legacy else 'full_executed_transition',
                        engine_terminated=nxt['terminated'], reward=reward, terminal=terminal,
                        truncated=truncated, scored_horizon=c.horizon, source='model', salt=prepared[j]['salt'],
                        contract=contract, temperature=1., decoding=DECODING, actor_sha256=actor_digest,
                        world_version=world_version, environment_version=environment_version,
                        balance_sha256=balance_sha, physical_slot=i)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    episodes[i]['decisions'] += 1
                    episodes[i]['source_counts']['model'] += 1
                    episodes[i]['actions'][action['verb']] += 1
                    states[i] = nxt
                    if terminal or truncated:
                        episodes[i].update(survival_ticks=final['tick'], terminal=terminal,
                                           alive_at_horizon=truncated, final=final)
                        reports.append(episodes[i])
                        finished.append(i)
                        print(json.dumps(dict(event='stochastic_episode', arm=arm,
                            seed=episodes[i]['seed'], replicate=episodes[i]['replicate'],
                            survival_ticks=final['tick'], alive_at_horizon=truncated)), flush=True)
                # Refill only after every decision in the physical batch is recorded.
                for slot in finished:
                    if schedule.complete(slot) is not None:
                        begin(slot)
        if policy.digest() != actor_digest:
            raise RuntimeError('Assessment changed actor weights')
    finally:
        for env in envs.values():
            env.close()
    reports.sort(key=lambda row: (row['replicate'], row['seed']))
    report = dict(arm=arm, profile=profile, world_version=world_version,
        environment_version=environment_version, balance_sha256=balance_sha,
        horizon=c.horizon, seeds=list(seeds), replicates=list(c.replicates),
        rng_prefix=c.rng_prefix, decoding=DECODING, temperature=1., contract=contract,
        scheduler='deterministic_dynamic_refill_fixed_physical_batch4',
        source_sha256=source_identity(c.project), episodes=reports, actor_sha256=actor_digest,
        trace_sha256=sha(trace), **summarize_episodes(reports, seeds, c.replicates, c.horizon))
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report



def score_evaluation_transition(tracker, transition, next_observation, horizon):
    from agent.rl_contract import transition_at_horizon
    # Potential requires the real observed successor. Reward is diagnostic;
    # survival scoring uses the exact engine tick inside a multi-tick action.
    reward = tracker.score(transition, next_observation)
    return reward, transition_at_horizon(transition, horizon)


def authorize_test(request):
    """Recompute the full integrity-reviewed selection before opening test worlds."""
    from agent.m8x_identity import sha,read_ledger
    from agent.m8x_registration import verify_registration
    from tools.compare_m8x_survival import build_result
    run=Path(request['run']).resolve()
    registration=json.loads((run/'registration.json').read_text(encoding='utf-8'))
    config=registration['config']
    if request.get('mode')!='test' or run!=Path(config['output']).resolve():
        raise ValueError('Only canonical registered held-out test is supported')
    verify_registration(config['project'],registration)
    if sha(run/'frozen_source.zip')!=registration['archive_sha256']:
        raise ValueError('Training archive changed')
    review_path=Path(request['selection_review'])
    review=json.loads(review_path.read_text(encoding='utf-8'))
    review_request=json.loads(Path(request['selection_review_request']).read_text(encoding='utf-8'))
    rebuilt=build_result(review_request,config['project'])
    if (review!=rebuilt or review.get('phase')!='selection' or review.get('decision')!='eligible_for_test'
            or review.get('evidence_complete') is not True or Path(review_request['training_run']).resolve()!=run
            or review.get('registration_sha256')!=sha(run/'registration.json')):
        raise ValueError('Selection review did not independently unlock this test')
    checkpoint=run/review['selected_checkpoint']
    if checkpoint.resolve().parent!=run or checkpoint.name not in ('checkpoint_0016','checkpoint_0032','checkpoint_0048'):
        raise ValueError('Selected checkpoint path differs')
    ledger=run/'ledger_test.jsonl'
    if ledger.exists():
        read_ledger(ledger)
        raise FileExistsError('Test phase already claimed; no retry or new output bypass')
    output=Path(request['output']).resolve()
    if output.exists() or output==run or output.parent!=run.parent:
        raise ValueError('Test output must be a new sibling of the canonical training run')
    return config,registration,review,checkpoint,ledger,output


def main():
    import argparse
    import gc
    from datetime import datetime,timezone
    from types import SimpleNamespace
    import torch
    from agent.m8x_identity import sha,execution_identity,append_ledger_event
    from agent.m8x_policy import Policy
    from agent.m6_policy import Policy as LegacyPolicy
    from agent.m8x_policy_identity import PARENT_ACTOR_SHA256
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    args=parser.parse_args();request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    config,registration,review,checkpoint,ledger,output=authorize_test(request)
    torch.set_num_threads(8)
    if execution_identity(config,torch)!=registration['execution_identity']:
        raise ValueError('Assessment execution identity differs from training')
    c=SimpleNamespace(**{**config,'replicates':config['test_replicates']})
    event=dict(attempt=c.attempt,phase='test',status='started',registration_sha256=sha(Path(request['run'])/'registration.json'),
               selected_checkpoint=checkpoint.name,selected_actor_sha256=review['selected_actor_sha256'],evidence_sha256={})
    append_ledger_event(ledger,event)
    output.mkdir()
    authorization=dict(request=request,registration_sha256=event['registration_sha256'],config=config,
                       execution_identity=registration['execution_identity'],source_sha256=registration['source_hashes'],
                       selection_review_sha256=sha(request['selection_review']),selected_checkpoint=str(checkpoint),
                       selected_actor_sha256=review['selected_actor_sha256'])
    (output/'authorization.json').write_text(json.dumps(authorization,indent=2),encoding='utf-8')
    try:
        for arm in ('original_v2','balanced_parent','candidate'):
            if arm=='original_v2':
                policy=LegacyPolicy(c.model,checkpoint=c.parent_checkpoint,seed=c.seed,max_tokens=c.max_tokens)
            else:
                policy=Policy(c.model,parent_checkpoint=c.parent_checkpoint,
                              checkpoint=checkpoint if arm=='candidate' else None,seed=c.seed,max_tokens=c.max_tokens,with_reference=False)
            expected=review['selected_actor_sha256'] if arm=='candidate' else PARENT_ACTOR_SHA256
            if policy.digest()!=expected:raise ValueError('Assessment actor identity changed')
            arm_config=SimpleNamespace(**{**vars(c),'profile':'open1' if arm=='original_v2' else 'open3'})
            evaluate(policy,arm_config,c.test_seeds,output/(arm+'.json'),arm=arm)
            del policy;gc.collect();torch.cuda.empty_cache()
        evidence={name+'.json':sha(output/(name+'.json')) for name in ('original_v2','balanced_parent','candidate')}
        append_ledger_event(ledger,{**event,'status':'completed','evidence_sha256':evidence})
        (output/'complete.json').write_text(json.dumps(dict(status='completed',mode='test',
            registration_sha256=event['registration_sha256'],evidence_sha256=evidence,
            finished_utc=datetime.now(timezone.utc).isoformat(),release_authorized=False),indent=2),encoding='utf-8')
    except BaseException as error:
        append_ledger_event(ledger,{**event,'status':'failed','error':type(error).__name__+': '+str(error)})
        raise


if __name__=='__main__':
    main()
