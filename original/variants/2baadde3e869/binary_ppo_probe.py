"""One real 16-row PPO update with simulated binary R2; no efficacy/hardware test."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def normalized(values):
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / max(sd, 1e-8) for v in values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--binary-package', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    # Keep the diagnostic's allocated GPUs entirely separate in time.
    status = json.loads((a.campaign / 'experiment01' / 'parallel_status.json').read_text())
    job = json.loads((a.campaign / 'job.json').read_text())
    if (status.get('complete') is not True or status.get('passed') is not True
            or status.get('accepted_cells') != 32 or job.get('accepted_complete') is not True):
        raise ValueError('Frozen diagnostic must finish successfully first')
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True).strip()
    if compute:
        raise RuntimeError('GPU compute is still active; do not run concurrently')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path[:0] = [str(a.campaign / 'project'), str(a.binary_package)]
    import torch
    from tools import mea_shuffle_train as train
    from agent.mea_shuffle import cell_config
    from agent.mea_replay import RecordingSnapshot
    from tools.mea_offline import evidence
    from mea_binary_r2.encoding import Encoder
    from mea_binary_r2.contract import TransitionGate
    from mea_binary_r2.offline import simulated_evidence

    raw = a.source.read_bytes()
    source_rows = [json.loads(line) for line in raw.decode().splitlines()]
    if len(source_rows) != 128:
        raise ValueError('Expected retained failed-run training rollout of 128 rows')
    rows = deepcopy(source_rows[:16])
    control = deepcopy(rows)
    config = train.CONFIG | dict(minibatch=16, critic_minibatch=16)
    write(a.output / 'registration.json', dict(
        technical_update_only=True, biological_effect_evaluated=False,
        hardware_commands_sent=False, r2_source='simulation', r2_enabled=True,
        state2_source='historical_28080_shuffled_recording', rows=16,
        source=str(a.source), source_sha256=hashlib.sha256(raw).hexdigest(),
        config=config, beta=.002, expected_actor_steps=2, expected_critic_steps=16,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        r2_fixture='alternating healthy and thirsty, independent of actual game reserves',
        weights_reused_in_campaign=False, efficacy_evaluated=False))
    policy = train.new_policy(a.campaign / 'model_config.json', cell_config(3, 'shuffled_after'))
    initial = train.state(policy)
    train.validate_common_initial(initial)
    prepared = policy.prepare([r['observation'] for r in rows],
                              [RecordingSnapshot.from_dict(r['snapshot']) for r in rows])
    for row, item in zip(rows, prepared):
        if evidence(item) != row['prepared']:
            raise ValueError('Exact historical input replay differs')
        row['prepared'] = item
    error = train.validate_policy_statistics(train.stats(policy, rows), rows)
    bootstrap = 0. if rows[-1]['terminal'] else source_rows[16]['value']
    bootstrap_error = 0.
    if not rows[-1]['terminal']:
        boot_row = source_rows[16]
        boot_input = policy.prepare([boot_row['observation']],
                                    [RecordingSnapshot.from_dict(boot_row['snapshot'])])[0]
        if evidence(boot_input) != boot_row['prepared']:
            raise ValueError('Exact historical bootstrap input differs')
        bootstrap_error = abs(train._value(policy, boot_input) - bootstrap)
        if bootstrap_error > 1e-6:
            raise ValueError('Historical bootstrap value differs from initial policy')
    train.finish_trajectory(control, bootstrap, config['gamma_tick'], config['gae_lambda'])
    train.audit_gae(control, bootstrap, config['gamma_tick'], config['gae_lambda'])
    binding = dict(session_id='SIMULATED-PPO-SMOKE', chip_id='SIMULATED-CHIP',
                   organoid_id='SIMULATED-ORGANOID', route_id='SIMULATED-ROUTE')
    gate = TransitionGate(binding)
    attached = []
    episode = 0
    for index, row in enumerate(rows):
        identity = binding | dict(episode_id=str(episode), transition_id=str(index),
                                  context_id='simulation-coverage-' + str(index))
        reserves = dict(H=100, F=100, W=100 if index % 2 == 0 else 80)
        intent = Encoder().decide(reserves, row['terminal'])
        receipt = None if row['terminal'] else simulated_evidence(
            identity, intent, 100000 + 400000 * index, 'event-' + str(index), 'window-' + str(index))
        reward = gate.accept(r1=row['reward'], duration=row['duration'],
            gamma=config['gamma_tick'], beta=.002, intent=intent, evidence=receipt,
            expected_identity=identity, terminal=row['terminal'])
        if not reward['valid'] or not math.isfinite(reward['total_reward']):
            raise ValueError('Simulated reward cannot enter PPO')
        row['original_reward_record'] = row['reward_record']
        row['reward_record'] = reward
        row['reward'] = reward['total_reward']
        attached.append(dict(index=index, fixture_reserves=reserves,
                             intent=asdict(intent), receipt=receipt, reward=reward))
        episode += int(row['terminal'])
    signs = [r['reward']['r2_raw'] for r in attached]
    if 1 not in signs or -1 not in signs:
        raise ValueError('Both positive and negative branches must reach this smoke')
    train.finish_trajectory(rows, bootstrap, config['gamma_tick'], config['gae_lambda'])
    train.audit_gae(rows, bootstrap, config['gamma_tick'], config['gae_lambda'])
    delta = [r['advantage'] - c['advantage'] for r, c in zip(rows, control)]
    normalized_delta = [x-y for x, y in zip(normalized([r['advantage'] for r in rows]),
                                             normalized([r['advantage'] for r in control]))]
    if max(map(abs, delta)) < 1e-8 or max(map(abs, normalized_delta)) < 1e-8:
        raise ValueError('Binary R2 did not change PPO advantages')
    serialized = deepcopy(rows)
    for row in serialized:
        row['prepared'] = evidence(row['prepared'])
    write(a.output / 'training_rows.json', serialized)
    write(a.output / 'simulation_receipts.json', attached)
    write(a.output / 'gae_comparison.json', dict(bootstrap=bootstrap,
        r1_advantages=[r['advantage'] for r in control],
        combined_advantages=[r['advantage'] for r in rows], delta=delta,
        normalized_delta=normalized_delta))
    opts = {name: torch.optim.AdamW(parameters, lr=lr, eps=1e-5, weight_decay=0)
            for name, parameters, lr in [
                ('actor', [p for p in policy.model.parameters() if p.requires_grad], config['actor_lr']),
                ('critic', policy.critic.parameters(), config['critic_lr'])]}
    torch.manual_seed(cell_config(3, 'shuffled_after')['ppo_seed'] + 1)
    metrics = train.training_update(policy, opts['actor'], opts['critic'], rows, SimpleNamespace(**config))
    counts = train.steps(opts)
    after = train.state(policy)
    if set(counts['actor']) != {2} or set(counts['critic']) != {16}:
        raise ValueError('Actual optimizer steps differ from technical budget')
    if (after['base'] != initial['base'] or after['reference'] != initial['reference']
            or after['actor'] == initial['actor'] or after['critic'] == initial['critic']):
        raise ValueError('Tensor update identity differs')
    receipt = dict(passed=True, technical_update_only=True, r2_enabled=True,
        r2_source='simulation', hardware_commands_sent=False, efficacy_evaluated=False,
        weights_saved=False, weights_reused_in_campaign=False, rows=len(rows),
        positive_rows=signs.count(1), negative_rows=signs.count(-1), terminal_rows=signs.count(None),
        initial=initial, after=after, pre_update_max_error=error, bootstrap_error=bootstrap_error,
        actor_steps=2, critic_steps=16, metrics=metrics,
        max_advantage_delta=max(map(abs, delta)),
        max_normalized_advantage_delta=max(map(abs, normalized_delta)),
        peak_cuda_allocated=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started)
    write(a.output / 'result.json', receipt)
    write(a.output / 'manifest.json', {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                      for f in sorted(a.output.iterdir()) if f.is_file()})
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
