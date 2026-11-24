"""Independent native Godot replay and CUDA categorical RNG reconstruction."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.m8s3_training import write, sha, verify_files
from agent.m8s3_env import GodotEnv
from agent.m8s3_contract import encode
from agent.m8s3_supply import create_reward as replay_reward
from agent.rl_contract import transition_at_horizon
from tools.audit_m8s3_replay import Comparator


def verify_generator_draws(generator, row, device):
    import torch
    if generator.get_state().cpu().numpy().tobytes().hex() != row['rng_before_hex']:
        raise ValueError('Episode RNG continuity differs from its registered initial seed or previous draw')
    v = int(torch.multinomial(torch.tensor(row['verb_probabilities'], device=device), 1, generator=generator))
    t = int(torch.multinomial(torch.tensor(row['target_probabilities'], device=device), 1, generator=generator))
    if v != row['verb_index'] or t != row['target_index']:
        raise ValueError('Sampler did not reproduce recorded action')
    if generator.get_state().cpu().numpy().tobytes().hex() != row['rng_after_hex']:
        raise ValueError('RNG successor differs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registration', required=True)
    ap.add_argument('--evaluation-only', action='store_true')
    ap.add_argument('--output', help='Separate destination for supplemental independent audits')
    a = ap.parse_args()
    reg = json.loads(Path(a.registration).read_text())
    c = reg['config']
    verify_files(c['project'], reg)
    out = Path(c['output'])
    comparison = Comparator(1e-12)
    counts = {}
    # Training rollouts preserve ongoing episodes across update boundaries.
    groups = [] if a.evaluation_only else [sorted(out.glob('rollout_*.jsonl.gz'))]
    groups += [[p] for p in sorted(out.glob('test_*.jsonl.gz' if a.evaluation_only else '*.jsonl.gz'))
               if not p.name.startswith('rollout_')]
    for files in groups:
        envs, states, trackers, seeds, generators = {}, {}, {}, {}, {}
        try:
            for path in files:
                count = 0
                endpoints = {}
                decisions = {}
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        slot, seed = row['env'], row['seed']
                        if slot not in envs:
                            envs[slot] = GodotEnv(c['godot'], c['project'])
                        if seeds.get(slot) != seed:
                            states[slot] = envs[slot].reset(seed)['observation']
                            trackers[slot] = replay_reward(states[slot], c)
                            seeds[slot] = seed
                            if 'rng_before_hex' in row:
                                import torch
                                from agent.m8_eval import episode_seed
                                generators[slot] = torch.Generator(device='cuda').manual_seed(
                                    episode_seed(c['rng_prefix'], 'open4', seed, 0))
                        comparison.same(states[slot], row['observation'], 'observation')
                        encoded = encode(states[slot])
                        chosen = encoded['branches'][row['verb_index']][row['target_index']]
                        comparison.same(chosen, row['action'], 'semantic_action')
                        nxt = envs[slot].step(row['action'])
                        comparison.same(nxt['transition'], row['transition'], 'transition')
                        comparison.same(nxt['observation'], row['next_observation'], 'successor')
                        reward = trackers[slot].score(nxt['transition'], nxt['observation'])
                        comparison.same(reward, row['reward'], 'reward')
                        states[slot] = nxt['observation']
                        if not path.name.startswith('rollout_'):
                            decisions[seed] = decisions.get(seed, 0) + 1
                            final = transition_at_horizon(nxt['transition'], 1800)['after']
                            if final['dead'] or final['tick'] == 1800:
                                if seed in endpoints:
                                    raise ValueError('Evaluation continued past its endpoint')
                                endpoints[seed] = dict(survival_ticks=final['tick'], terminal=final['dead'],
                                    alive_at_horizon=not final['dead'], decisions=decisions[seed])
                        if 'rng_before_hex' in row:
                            verify_generator_draws(generators[slot], row, 'cuda')
                        count += 1
                if not path.name.startswith('rollout_'):
                    report = json.loads(path.with_suffix('').with_suffix('.json').read_text())
                    if len(report['episodes']) != len(endpoints) or {e['seed'] for e in report['episodes']} != set(endpoints):
                        raise ValueError('Evaluation summary is not the complete replayed world matrix')
                    for episode in report['episodes']:
                        for key, actual in endpoints[episode['seed']].items():
                            comparison.same(actual, episode[key], 'episode_summary.' + key)
                counts[path.name] = dict(transitions=count, sha256=sha(path))
                print(json.dumps(dict(audited=path.name, transitions=count)), flush=True)
        finally:
            for env in envs.values():
                env.close()
    destination = Path(a.output) if a.output else out / ('independent_test_replay.json' if a.evaluation_only else 'independent_replay.json')
    write(destination, dict(passed=True, files=counts, auditor_sha256=sha(Path(__file__)),
          transitions=sum(r['transitions'] for r in counts.values()), max_float_error=comparison.max_float_error,
          registration_sha256=reg['registration_sha256']))


if __name__ == '__main__':
    main()
