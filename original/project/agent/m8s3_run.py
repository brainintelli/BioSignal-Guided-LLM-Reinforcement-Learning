"""Registered open4 PPO pilot/training with independent stochastic assessment.

No curriculum, action override, test feedback, or training time-limit terminal.
GPU ownership belongs to this finite process, supervised by m8s3_job.py.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

from agent.m8s3_training import (sha, write, verify_files, validate_seed_split,
                                selection_eligible, claim_test, authorize_test, should_stop_training, training_slots)
from tools.m8b_metrics import summarize, compare_test
from agent.m8s3_supply import create_reward as training_reward


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def record(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def exploration_distribution(base, mask, epsilon):
    """One categorical policy, not a post-sampling action replacement.

    Both PPO and the independent evaluator use this exact differentiable mixture.
    The uniform component sees legality only, never reserves or preferred actions.
    """
    import torch
    if not 0 <= epsilon < 1:
        raise ValueError('Exploration epsilon must be in [0,1)')
    if epsilon == 0:
        return base
    legal = mask.to(dtype=base.probs.dtype, device=base.probs.device)
    if not bool((legal.sum(-1) > 0).all()):
        raise ValueError('No legal verb')
    uniform = legal / legal.sum(-1, keepdim=True)
    return torch.distributions.Categorical(probs=(1-epsilon)*base.probs + epsilon*uniform)


@contextmanager
def preserve_rng():
    import torch
    cpu, gpu, py = torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(gpu)
        random.setstate(py)


def load_checkpoint(policy, checkpoint):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(checkpoint / name) != digest:
            raise ValueError('Checkpoint file changed: ' + name)
    meta = json.loads((checkpoint / 'policy.json').read_text())
    if meta['inference_layout'] != 'semantic_v3_canonical_state_buckets_b4':
        raise ValueError('Incompatible checkpoint layout')
    current = policy.metadata()
    if meta.get('verb_exploration_epsilon', 0) != current.get('verb_exploration_epsilon', 0):
        raise ValueError('Checkpoint exploration policy differs')
    for name in ('contract', 'balance_sha256', 'world_version', 'base_config_sha256',
                 'reward_config', 'kernel_identity', 'critic_feature_names'):
        if meta[name] != current[name]:
            raise ValueError('Checkpoint identity differs: ' + name)
    set_peft_model_state_dict(policy.model, load_file(str(checkpoint / 'adapter/adapter_model.safetensors')))
    policy.load_critic_state(torch.load(checkpoint / 'critic.pt', map_location=policy.device_name,
                                       weights_only=True), meta)
    policy.execution_cache.revisions.clear()
    policy.clear_prefix_cache()
    if policy.digest() != meta['actor_sha256']:
        raise ValueError('Restored actor digest mismatch')
    return manifest


def save_checkpoint(policy, destination, actor_optimizer, critic_optimizer, update, registration_digest):
    import torch
    if len(registration_digest) != 64 or any(ch not in '0123456789abcdef' for ch in registration_digest):
        raise ValueError('An exact training registration digest is required')
    meta = policy.metadata()
    if (meta['inference_layout'] != 'semantic_v3_canonical_state_buckets_b4'
            or meta['contract'] != 'open_world_semantic_survival_v3'):
        raise ValueError('Only the registered new-world policy may be saved')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Never replace a completed checkpoint')
    staging = destination.with_name(destination.name + '.partial')
    staging.mkdir(exist_ok=False)
    # The inherited FastPolicy.save deliberately rejects diagnostic layouts.
    # This independently registered writer owns the new-world checkpoint format.
    policy.model.save_pretrained(staging / 'adapter', safe_serialization=True)
    torch.save(policy.critic.state_dict(), staging / 'critic.pt')
    meta.update(formal_training_started=True, checkpoint_writes=1, release_authorized=False,
                registration_sha256=registration_digest, update=update)
    write(staging / 'policy.json', meta)
    torch.save(dict(actor=actor_optimizer.state_dict(), critic=critic_optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate()), staging / 'optimizers.pt')
    files = {str(p.relative_to(staging)): sha(p) for p in staging.rglob('*') if p.is_file()}
    write(staging / 'manifest.json', dict(update=update, files=files,
          actor_sha256=policy.digest(), registration_sha256=registration_digest,
          resume_semantics='optimizer state retained for audit; no implicit episode restart/resume'))
    staging.rename(destination)


def evaluate(policy, c, seeds, destination, *, reference=False):
    import torch
    from agent.m8_eval import episode_seed, summarize_episodes
    from agent.m8x_eval import SlotSchedule
    from agent.m8s3_env import GodotEnv
    from agent.m8s3_contract import validate_observation_identity
    from agent.rl_contract import transition_at_horizon
    destination = Path(destination)
    trace = destination.with_suffix('.jsonl.gz')
    if destination.exists() or trace.exists():
        raise FileExistsError('Assessment evidence already exists')
    schedule = SlotSchedule([(seed, 0) for seed in seeds])
    envs, states, trackers, generators, episodes = {}, {}, {}, {}, {}
    reports = []
    actor = policy.reference_digest() if reference else policy.digest()
    began = time.monotonic()

    def begin(slot):
        seed, rep = schedule.active[slot]
        states[slot] = envs[slot].reset(seed, 'open4')
        validate_observation_identity(states[slot]['observation'])
        trackers[slot] = training_reward(states[slot]['observation'], vars(c))
        rng = episode_seed(c.rng_prefix, 'open4', seed, rep)
        generators[slot] = torch.Generator(device=policy.device_name).manual_seed(rng)
        episodes[slot] = dict(seed=seed, replicate=rep, rng_seed=rng, decisions=0, actions=Counter())

    try:
        for slot in sorted(schedule.active):
            envs[slot] = GodotEnv(c.godot, c.project)
            begin(slot)
        with torch.no_grad(), gzip.open(trace, 'xt', encoding='utf-8', compresslevel=1) as stream:
            while schedule.active:
                active = sorted(schedule.active)
                prepared = policy.prepare([states[i]['observation'] for i in active])
                with policy._scope(prepared, reference):
                    vd = policy.verb_distribution(prepared, reference)
                    before = [generators[i].get_state().cpu().numpy().tobytes().hex() for i in active]
                    verbs = [int(torch.multinomial(vd.probs[j], 1, generator=generators[i])) for j, i in enumerate(active)]
                    td = policy.target_distribution(prepared, verbs, reference)
                    targets = [int(torch.multinomial(td.probs[j], 1, generator=generators[i])) for j, i in enumerate(active)]
                finished = []
                for j, i in enumerate(active):
                    action = prepared[j]['branches'][verbs[j]][targets[j]]
                    nxt = envs[i].step(action)
                    reward = trackers[i].score(nxt['transition'], nxt['observation'])
                    scored = transition_at_horizon(nxt['transition'], 1800)['after']
                    terminal = scored['dead']
                    alive = scored['tick'] == 1800 and not terminal
                    record(stream, dict(seed=episodes[i]['seed'], env=i, observation=states[i]['observation'],
                        action=action, transition=nxt['transition'], next_observation=nxt['observation'],
                        reward=reward, source='model', actor_sha256=actor,
                        verb_index=verbs[j], target_index=targets[j],
                        verb_probabilities=vd.probs[j].tolist(), target_probabilities=td.probs[j].tolist(),
                        logp=float(vd.log_prob(torch.tensor(verbs, device=policy.device_name))[j]
                                   + td.log_prob(torch.tensor(targets, device=policy.device_name))[j]),
                        rng_before_hex=before[j],
                        rng_after_hex=generators[i].get_state().cpu().numpy().tobytes().hex()))
                    episodes[i]['decisions'] += 1
                    episodes[i]['actions'][action['verb']] += 1
                    states[i] = nxt
                    if terminal or alive:
                        episodes[i].update(survival_ticks=scored['tick'], terminal=terminal,
                                           alive_at_horizon=alive, final=scored)
                        reports.append(episodes[i])
                        emit(event='evaluation_episode', evaluation=destination.stem, seed=episodes[i]['seed'],
                             survival_ticks=scored['tick'], alive_at_horizon=alive)
                        finished.append(i)
                for slot in finished:
                    if schedule.complete(slot) is not None:
                        begin(slot)
        after = policy.reference_digest() if reference else policy.digest()
        if after != actor:
            raise RuntimeError('Assessment changed the policy')
    finally:
        for env in envs.values():
            env.close()
    reports.sort(key=lambda r: r['seed'])
    summary = summarize_episodes(reports, seeds, [0], 1800)
    result = dict(episodes=reports, summary=summarize(reports), **summary,
                  actor_sha256=actor, seconds=time.monotonic()-began,
                  decisions=sum(x['decisions'] for x in reports), trace_sha256=sha(trace),
                  profile='open4', world_version=6, temperature=1, reference=reference)
    write(destination, result)
    return result


def make_policy(c, reference=True):
    import torch
    from agent.m8s3_batch_policy import CanonicalBatchPolicy
    epsilon = float(getattr(c, 'verb_exploration_epsilon', 0))
    class TrainingPolicy(CanonicalBatchPolicy):
        readiness_enabled = False
        optimization_anchor_active = False
        reward_installed = False

        def verb_distribution(self, prepared, reference=False):
            base = super().verb_distribution(prepared, reference)
            mask = torch.tensor([r['verb_mask'] for r in prepared], device=self.device_name)
            return exploration_distribution(base, mask, epsilon)

        def values(self, prepared):
            if not self.readiness_enabled:
                return super().values(prepared)
            from agent.m8s3_supply import readiness_features
            facts=torch.tensor([r['features']+readiness_features(r) for r in prepared],device=self.device_name,dtype=torch.float32)
            return self.critic(facts).squeeze(-1)

        def load_critic_state(self,state_dict,metadata):
            if not self.readiness_enabled:
                return super().load_critic_state(state_dict,metadata)
            expected=self.metadata()
            if (metadata.get('critic_features')!=108 or metadata.get('critic_feature_names')!=expected['critic_feature_names']
                    or any(metadata.get(k)!=v for k,v in self.world_identity.items())
                    or tuple(state_dict['0.weight'].shape)!=(128,108)):
                raise ValueError('Readiness critic identity or shape differs')
            self.critic.load_state_dict(state_dict,strict=True)
            self.critic_initialization='explicit108_feature_checkpoint'

        def assert_reference_frozen(self):
            if not self.optimization_anchor_active:
                return super().assert_reference_frozen()
            if (self.reference_model is None or any(p.requires_grad or p.grad is not None for p in self.reference_model.parameters())
                    or self.reference_digest()!=c.optimization_anchor_actor_sha256):
                raise RuntimeError('Frozen optimization anchor changed')

        def metadata(self):
            result = super().metadata()
            result.update(verb_exploration_epsilon=epsilon,
                verb_policy='(1-epsilon)*LM categorical + epsilon*uniform over all legal verbs; one categorical draw; same sampler/loss/eval/reference')
            if self.reward_installed:
                from agent.m8s3_supply import reward_config
                result['reward_config']=reward_config(vars(c))
            if self.readiness_enabled:
                from agent.m8s3_supply import READINESS_NAMES
                result['critic_features']=108
                result['critic_feature_names']+=list(READINESS_NAMES)
            if self.optimization_anchor_active:
                result['reference_actor_sha256']=c.optimization_anchor_actor_sha256
                result['optimization_reference_role']='unreleased warm candidate; evaluation parent remains released M6'
            return result
    policy = TrainingPolicy(c.model, checkpoint=c.initial_checkpoint,
        parent_checkpoint=c.parent_checkpoint, seed=c.rng_seed, reset_critic=True,
        with_reference=reference, max_tokens=1536, prefix_layout='system_only')
    if getattr(c, 'warm_start_checkpoint', None):
        # A new registered experiment, not an optimizer/episode resume. The old
        # incompatible critic was never loaded; only this explicit 96-D state is.
        if not c.warm_start_critic or c.reset_critic:
            raise ValueError('Warm-start initialization declaration differs')
        load_checkpoint(policy, c.warm_start_checkpoint)
    if getattr(c,'critic_readiness',False):
        from agent.m8s3_supply import expand_critic
        expand_critic(policy.critic)
        policy.readiness_enabled=True
        policy.critic_initialization='warm96_features_plus12_zero_initialized_readiness_columns'
    policy.reward_installed=True
    return policy


def activate_optimization_anchor(policy,c,out):
    if not getattr(c,'optimization_anchor_checkpoint',None):return
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    checkpoint=Path(c.optimization_anchor_checkpoint)
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(checkpoint/name)!=digest:raise ValueError('Optimization anchor input changed')
    if manifest['actor_sha256']!=c.optimization_anchor_actor_sha256:
        raise ValueError('Optimization anchor actor differs')
    previous=policy.reference_digest()
    set_peft_model_state_dict(policy.reference_model,load_file(str(checkpoint/'adapter/adapter_model.safetensors')))
    policy.reference_model.requires_grad_(False);policy.reference_model.eval()
    policy.optimization_anchor_active=True
    policy.execution_cache.revisions.clear();policy.clear_prefix_cache()
    policy.assert_reference_frozen()
    write(out/'optimization_anchor.json',dict(evaluation_parent_actor_sha256=previous,
        optimization_anchor_actor_sha256=policy.reference_digest(),checkpoint=str(checkpoint),
        evaluation_parent_unchanged=True,scope='Training KL anchor only; not a released parent or test baseline'))


def train(c, reg, out):
    import torch
    from agent.m8s3_env import GodotEnv
    from agent.m8x_train import finish_trajectory
    from agent.m8_train import update
    from tools.benchmark_m8s3 import selected_logps
    policy = make_policy(c)
    initial_actor = policy.digest()
    actor_optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],
                                        lr=c.actor_lr, eps=1e-5, weight_decay=0)
    critic_optimizer = torch.optim.AdamW(policy.critic.parameters(), lr=c.critic_lr, eps=1e-5, weight_decay=0)
    baseline_seeds = c.pilot_eval_seeds if c.mode == 'pilot' else c.validation_seeds
    with preserve_rng():
        initial_eval = evaluate(policy, c, baseline_seeds, out / 'initial_evaluation.json')
        parent_eval = None if c.mode == 'pilot' else evaluate(policy, c, baseline_seeds, out / 'parent_selection.json', reference=True)
    activate_optimization_anchor(policy,c,out)
    envs, states, trackers, seeds = [], [], [], []
    cursor = 0
    def reset(i):
        nonlocal cursor
        seed = c.train_seed_start + cursor % c.train_seed_count
        cursor += 1
        state = envs[i].reset(seed)
        return state, training_reward(state['observation'],vars(c)), seed
    random.seed(c.rng_seed)
    torch.manual_seed(c.rng_seed)
    began = time.monotonic()
    candidates, timings = [], []
    total_ticks = total_decisions = 0
    try:
        pool_size = getattr(c, 'environment_pool_size', c.envs)
        training_slots(1, c.envs, pool_size)
        for _ in range(pool_size):
            envs.append(GodotEnv(c.godot, c.project))
        for i in range(pool_size):
            state, tracker, seed = reset(i)
            states.append(state); trackers.append(tracker); seeds.append(seed)
        for number in range(1, c.updates + 1):
            if time.monotonic() - began > c.max_train_seconds:
                raise TimeoutError('Registered training budget exhausted')
            started = time.monotonic()
            active = training_slots(number, c.envs, pool_size)
            trajectories = [[] for _ in active]
            counts = Counter()
            with gzip.open(out / f'rollout_{number:04d}.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as stream:
                for step in range(c.rollout_steps):
                    prepared = policy.prepare([states[slot]['observation'] for slot in active])
                    sampled = policy.act(prepared, with_reference=True)
                    for i, slot in enumerate(active):
                        env = envs[slot]
                        action = sampled['actions'][i]
                        nxt = env.step(action)
                        scored = trackers[slot].score(nxt['transition'], nxt['observation'])
                        terminal = nxt['terminated']
                        row = dict(prepared=prepared[i], verb=sampled['verbs'][i], target=sampled['targets'][i],
                            value=sampled['values'][i], logp=sampled['logp'][i], verb_logp=sampled['verb_logp'][i],
                            reference_verb=sampled['reference_verb'][i], reference_target=sampled['reference_target'][i],
                            reward=scored['reward'], duration=scored['elapsed_ticks'], terminal=terminal,
                            boundary=terminal, next_obs=nxt['observation'])
                        trajectories[i].append(row)
                        counts[action['verb']] += 1
                        total_decisions += 1
                        total_ticks += row['duration']
                        record(stream, dict(update=number, step=step, env=slot, seed=seeds[slot], source='model',
                            observation=states[slot]['observation'], action=action, transition=nxt['transition'],
                            next_observation=nxt['observation'], reward=scored, logp=row['logp'],
                            verb_index=row['verb'], target_index=row['target'], input_ids=prepared[i]['input_ids']))
                        if terminal:
                            with (out / 'episodes.jsonl').open('a', encoding='utf-8') as es:
                                record(es, dict(update=number, seed=seeds[slot], survival_ticks=nxt['observation']['tick'],
                                                final=nxt['transition']['after']))
                            states[slot], trackers[slot], seeds[slot] = reset(slot)
                        else:
                            states[slot] = nxt
            rows = []
            for trajectory in trajectories:
                with torch.no_grad():
                    value = 0 if trajectory[-1]['terminal'] else float(policy.values(policy.prepare([trajectory[-1]['next_obs']]))[0])
                rows.extend(finish_trajectory(trajectory, value, c.gamma_tick, c.gae_lambda))
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - started
            lp = selected_logps(policy, [r['prepared'] for r in rows], [r['verb'] for r in rows], [r['target'] for r in rows])
            error = max(abs(x-r['logp']) for x, r in zip(lp, rows))
            if error > 1e-6:
                raise RuntimeError('All-row pre-update probability drift: ' + str(error))
            metrics = update(policy, actor_optimizer, critic_optimizer, rows, c)
            torch.cuda.synchronize()
            timing = dict(update=number, decisions=total_decisions, ticks=total_ticks,
                active_environment_slots=active, environment_pool_size=pool_size,
                rollout_seconds=rollout_seconds, update_seconds=time.monotonic()-started-rollout_seconds,
                pre_update_all_logp_error=error, actions=dict(counts), **metrics)
            timings.append(timing)
            with (out / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                record(stream, timing)
            emit(event='training_update', **timing)
            write(out / 'status.json', dict(status='training', latest=timing, target_completed=False))
            if number % c.save_every == 0 or number == c.updates:
                checkpoint = out / f'checkpoint_{number:04d}'
                save_started = time.monotonic()
                save_checkpoint(policy, checkpoint, actor_optimizer, critic_optimizer, number, reg['registration_sha256'])
                # Check the actual saved adapter and 96-feature critic in process;
                # a second process independently checks selected checkpoints later.
                with preserve_rng():
                    checks = [r['prepared'] for r in rows[:16]]
                    vs, ts = [r['verb'] for r in rows[:16]], [r['target'] for r in rows[:16]]
                    before = selected_logps(policy, checks, vs, ts)
                    values = policy.values(checks).detach().cpu()
                    load_checkpoint(policy, checkpoint)
                    after = selected_logps(policy, checks, vs, ts)
                    reload_error = max(abs(a-b) for a, b in zip(before, after))
                    critic_error = float((values-policy.values(checks).detach().cpu()).abs().max())
                    if reload_error > 1e-6 or critic_error != 0:
                        raise ValueError('Checkpoint roundtrip differs')
                    write(checkpoint / 'reload_probe.json', dict(prepared_messages=[r['messages'] for r in checks],
                        # Input observations below come from the saved rollout, selected by audit.
                        verbs=vs, targets=ts, expected_logp=before, logp_error=reload_error, critic_error=critic_error))
                emit(event='checkpoint_saved', checkpoint=checkpoint.name, seconds=time.monotonic()-save_started,
                     logp_reload_error=reload_error, critic_reload_error=critic_error)
            if number % c.validation_every == 0 or number == c.updates:
                with preserve_rng():
                    result = evaluate(policy, c, baseline_seeds, out / f'selection_{number:04d}.json')
                candidates.append((number, result))
                write(out / 'selection_progress.json', [dict(update=n, summary=r['summary']) for n, r in candidates])
                if c.mode == 'formal' and should_stop_training(candidates, parent_eval['episodes'], vars(c)):
                    emit(event='registered_training_stop', update=number, reason='consecutive_perfect_screens')
                    break
    finally:
        for env in envs:
            env.close()
    if policy.digest() == initial_actor:
        raise RuntimeError('No actor learning occurred')
    write(out / 'training_complete.json', dict(updates=number, registered_maximum_updates=c.updates,
        stopping_reason='registered_consecutive_perfect' if number < c.updates else 'registered_budget_exhausted',
        decisions=total_decisions, ticks=total_ticks,
        seconds=time.monotonic()-began, initial_actor=initial_actor, final_actor=policy.digest(),
        initial_evaluation=initial_eval['summary'], final_evaluation=candidates[-1][1]['summary']))
    if c.mode == 'pilot':
        write(out / 'status.json', dict(status='pilot_completed', target_completed=False))
        return
    winner, result = max(candidates, key=lambda pair: (pair[1]['summary']['successes'],
        pair[1]['summary']['cvar20'], pair[1]['summary']['rmst'], -pair[0]))
    selected = out / f'checkpoint_{winner:04d}'
    eligible = selection_eligible(result['episodes'], parent_eval['episodes'])
    write(out / 'selection.json', dict(update=winner, checkpoint=str(selected), eligible_for_test=eligible,
                                     summary=result['summary'], actor_sha256=result['actor_sha256']))
    if not eligible:
        write(out / 'status.json', dict(status='selection_failed', target_completed=False,
                                       reason='No candidate met the frozen 15/16 and noninferiority screen'))
        return
    write(out / 'status.json', dict(status='selected_pending_independent_audit', target_completed=False,
                                   selected_checkpoint=str(selected)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registration', required=True)
    ap.add_argument('--reload-checkpoint')
    ap.add_argument('--legacy-test', action='store_true')
    ap.add_argument('--test-stage', action='store_true')
    a = ap.parse_args()
    reg = json.loads(Path(a.registration).read_text())
    from agent.m8s3_diag_contract import verify_registration_digest
    verify_registration_digest(reg)
    c = SimpleNamespace(**reg['config'])
    verify_files(c.project, reg)
    validate_seed_split(list(range(c.train_seed_start, c.train_seed_start+c.train_seed_count)),
                        c.validation_seeds, c.test_seeds, c.pilot_eval_seeds)
    if c.profile != 'open4' or c.horizon != 1800 or c.temperature != 1:
        raise ValueError('Unregistered world or evaluation settings')
    training_slots(1,c.envs,getattr(c,'environment_pool_size',c.envs))
    from tools.benchmark_m8s3 import setup
    setup()
    out = Path(c.output)
    if a.reload_checkpoint:
        reload_audit(c, reg, Path(a.reload_checkpoint))
    elif a.legacy_test:
        legacy_test(c, out)
    elif a.test_stage:
        selected = authorize_test(out, reg['config'])
        policy = make_policy(c)
        load_checkpoint(policy, selected)
        claim_test(out, policy.digest())
        candidate = evaluate(policy, c, c.test_seeds, out / 'test_candidate.json')
        parent = evaluate(policy, c, c.test_seeds, out / 'test_parent.json', reference=True)
        write(out / 'status.json', dict(status='test_completed_pending_audit', target_completed=False,
              candidate=candidate['summary'], parent=parent['summary'], selected_checkpoint=str(selected)))
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'registration.json', reg)
        try:
            train(c, reg, out)
            verify_files(c.project, reg)
        except BaseException as error:
            write(out / 'failure.json', dict(type=type(error).__name__, error=str(error)))
            write(out / 'status.json', dict(status='failed', target_completed=False, error=str(error)))
            raise


def reload_audit(c, reg, checkpoint):
    from tools.benchmark_m8s3 import selected_logps
    policy = make_policy(c, reference=False)
    load_checkpoint(policy, checkpoint)
    probe = json.loads((checkpoint / 'reload_probe.json').read_text())
    # Training rows are env-major after GAE; reconstruct matching first 16 rows.
    update = json.loads((checkpoint / 'manifest.json').read_text())['update']
    with gzip.open(Path(c.output) / f'rollout_{update:04d}.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    rows = sorted(rows, key=lambda r: (r['env'], r['step']))[:16]
    prepared = policy.prepare([r['observation'] for r in rows])
    if [p['messages'] for p in prepared] != probe['prepared_messages']:
        raise ValueError('Checkpoint audit observations differ')
    lp = selected_logps(policy, prepared, probe['verbs'], probe['targets'])
    error = max(abs(x-y) for x, y in zip(lp, probe['expected_logp']))
    if error > 1e-6:
        raise ValueError('Independent-process checkpoint probability mismatch: ' + str(error))
    write(checkpoint / 'independent_reload.json', dict(passed=True, logp_error=error,
          actions=len(rows), actor_sha256=policy.digest(), registration_sha256=reg['registration_sha256']))
    emit(event='independent_reload_passed', logp_error=error)


def legacy_test(c, out):
    from agent.m6_policy import Policy
    from agent.m8x_eval import evaluate as legacy_evaluate
    from agent.m8s2_kernels import pin_kernels
    pin_kernels()
    if not (out / 'test_candidate.json').exists():
        raise ValueError('Held-out candidate must finish before the descriptive legacy arm')
    policy = Policy(c.model, checkpoint=c.parent_checkpoint, max_tokens=1536)
    legacy = SimpleNamespace(**vars(c))
    legacy.profile = 'open1'
    legacy.eval_batch_size = 4
    legacy.replicates = [0]
    old = legacy_evaluate(policy, legacy, c.test_seeds, out / 'test_original_v2.json', arm='original_v2')
    candidate = json.loads((out / 'test_candidate.json').read_text())
    parent = json.loads((out / 'test_parent.json').read_text())
    result = compare_test(candidate['episodes'], parent['episodes'], old['episodes'])
    write(out / 'numerical_gate.json', result)
    write(out / 'status.json', dict(status='numerical_pass_pending_audit_and_online' if result['numerical_gate_passed']
                                   else 'numerical_gate_failed', target_completed=False,
                                   numerical_gate_passed=result['numerical_gate_passed']))


if __name__ == '__main__':
    main()
