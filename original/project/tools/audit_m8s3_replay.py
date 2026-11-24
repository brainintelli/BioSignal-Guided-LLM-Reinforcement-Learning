"""Independently replay recorded GPU actions in the local native Godot engine.

No model, GPU, optimizer or synthetic environment is loaded. The navigation
records omit seeds/raw observations, so their fixed seeds come from the probe
generator and that evidence limitation is explicitly recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.m8s3_contract import encode, SurvivalReward, VERBS, balance_identity
from agent.m8s3_env import GodotEnv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Comparator:
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.float_values = 0
        self.max_float_error = 0.

    def same(self, actual, expected, path):
        if isinstance(expected, dict):
            assert isinstance(actual, dict) and set(actual) == set(expected), f'{path}: object keys differ'
            for key in expected:
                self.same(actual[key], expected[key], f'{path}.{key}')
        elif isinstance(expected, list):
            assert isinstance(actual, list) and len(actual) == len(expected), f'{path}: array length differs'
            for index, (a, b) in enumerate(zip(actual, expected)):
                self.same(a, b, f'{path}[{index}]')
        elif type(expected) in (float, int) and type(actual) in (float, int):
            assert math.isfinite(actual) and math.isfinite(expected), f'{path}: nonfinite number'
            if isinstance(expected, float) or isinstance(actual, float):
                difference = abs(actual - expected)
                self.float_values += 1
                self.max_float_error = max(self.max_float_error, difference)
                assert difference <= self.tolerance, f'{path}: {actual!r} != {expected!r}'
            else:
                assert actual == expected, f'{path}: integer differs'
        else:
            assert type(actual) is type(expected) and actual == expected, f'{path}: value differs'


def no_energy(value, path):
    if isinstance(value, dict):
        assert 'energy' not in value, f'{path}: retired reserve leaked'
        for key, item in value.items():
            no_energy(item, f'{path}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            no_energy(item, f'{path}[{index}]')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--godot', type=Path, default=ROOT/'.tools/godot/Godot_v4.7.2-stable_win64_console.exe')
    parser.add_argument('--probe', type=Path, default=ROOT/'output/rules_v2_20261108/remote_evidence/probe_default.json')
    parser.add_argument('--reload', type=Path, default=ROOT/'output/rules_v2_20261108/remote_evidence/reload_default.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/rules_v2_20261108/replay_audit.json')
    parser.add_argument('--rollout-seed', type=int, default=883810)
    parser.add_argument('--navigation-seed-start', type=int, default=883810)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args()
    args.godot, args.project = args.godot.resolve(), args.project.resolve()
    assert args.godot.is_file(), 'A real installed Godot executable is required'
    assert 0 <= args.tolerance <= 1e-12, 'Do not relax numeric audit tolerance'
    version = subprocess.run([str(args.godot), '--version'], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    assert version.startswith('4.'), 'Expected native Godot 4'
    probe = json.loads(args.probe.read_text(encoding='utf-8'))
    reload = json.loads(args.reload.read_text(encoding='utf-8'))
    assert probe['rollout_decisions'] == len(probe['rollout']) > 0
    for evidence in (probe, reload):
        assert evidence['actor_unchanged'] and evidence['optimizer_steps'] == 0
        assert evidence['checkpoint_writes'] == 0 and not evidence['formal_training_started']
    assert reload['independent_process'] and reload['actions_checked'] == len(probe['rollout'])
    assert reload['logp_error_max'] <= 1e-6
    comparison = Comparator(args.tolerance)
    identity = balance_identity(args.project)
    for key, value in identity.items():
        assert probe['metadata'][key] == value and reload['metadata'][key] == value, key
    counts = Counter()
    actions = Counter()
    details = []

    def replay_record(env, observation, reward, record, label, seed):
        no_energy(observation, label+'.observation')
        no_energy(record['transition'], label+'.transition')
        if 'observation' in record:
            comparison.same(observation, record['observation'], label+'.observation')
            counts['raw_observations_compared'] += 1
        row = encode(observation)
        comparison.same(row['messages'], record['state'], label+'.model_input')
        counts['model_inputs_compared'] += 1
        assert len(row['features']) == 96
        assert record['action'] in observation['allowed_actions'], label+': action not legal'
        verb = VERBS.index(record['action']['verb'])
        target = row['branches'][verb].index(record['action'])
        comparison.same(row['target_texts'][verb][target], record['semantic_target'], label+'.semantic_target')
        response = env.step(record['action'])
        no_energy(response, label+'.response')
        comparison.same(response['transition'], record['transition'], label+'.transition')
        comparison.same(response['terminated'], record['transition']['after']['dead'], label+'.terminated')
        computed_reward = reward.score(response['transition'], response['observation'])
        comparison.same(computed_reward, record['reward'], label+'.reward')
        counts['complete_transitions_compared'] += 1
        counts['reward_records_compared'] += 1
        counts['exact_tick_snapshots_compared'] += len(response['transition']['trace'])
        actions[record['action']['verb']] += 1
        if record['action']['verb'] in ('move', 'navigate'):
            before, after = (response['transition'][key]['player'] for key in ('before', 'after'))
            assert abs(after['x']-before['x']) + abs(after['y']-before['y']) == 1, label+': not one tile'
            assert response['transition']['elapsed_ticks'] == 1, label+': not one tick'
            counts['one_tile_one_tick_actions'] += 1
        details.append(dict(record=label, seed=seed, action=record['action'], semantic_target=record['semantic_target'],
                            elapsed_ticks=response['transition']['elapsed_ticks'], reward=computed_reward['reward']))
        return response['observation']

    with GodotEnv(str(args.godot), str(args.project)) as env:
        observation = env.reset(args.rollout_seed)['observation']
        reward = SurvivalReward(observation)
        for step, record in enumerate(probe['rollout']):
            assert record['step'] == step, 'Rollout order changed'
            observation = replay_record(env, observation, reward, record, f'rollout[{step}]', args.rollout_seed)
    for index, record in enumerate(probe['navigation_probe']):
        seed = args.navigation_seed_start + index
        assert record['source'] == 'forced_navigation_verb_model_selects_target'
        assert record['action']['verb'] == 'navigate'
        with GodotEnv(str(args.godot), str(args.project)) as env:
            observation = env.reset(seed)['observation']
            replay_record(env, observation, SurvivalReward(observation), record, f'forced_navigation[{index}]', seed)
    source_names = ['agent/m8s3_contract.py', 'agent/m8s3_env.py', 'scripts/survival_world.gd',
                    'scripts/m8s3_environment_session.gd', 'scripts/world_state.gd',
                    'scripts/rl_environment_session.gd', 'configs/m8_survival_v2.json',
                    'godot_tests/m8s3_environment.gd', 'tools/probe_m8s3.py', 'tools/audit_m8s3_replay.py']
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  engine=dict(executable=str(args.godot), sha256=sha(args.godot), version=version,
                              transport='agent.m8s3_env.GodotEnv native subprocess', synthetic_fallback=False),
                  identity=identity, probe_sha256=sha(args.probe), reload_sha256=sha(args.reload),
                  source_sha256={name:sha(args.project/name) for name in source_names},
                  tolerance=args.tolerance, float_values_compared=comparison.float_values,
                  max_float_difference=comparison.max_float_error, counts=dict(counts), action_counts=dict(actions),
                  rollout_seed=args.rollout_seed, navigation_seeds=list(range(args.navigation_seed_start, args.navigation_seed_start+len(probe['navigation_probe']))),
                  seed_provenance='Fixed seed literals in tools/probe_m8s3.py; source hash recorded',
                  evidence_boundary='Navigation JSON omits raw observation/seed; reconstructed input messages and complete transitions were compared. No model logits were recomputed by this CPU audit.',
                  independent_gpu_reload_report=dict(actions_checked=reload['actions_checked'], logp_error_max=reload['logp_error_max']),
                  records=details, model_loaded=False, optimizer_steps=0, checkpoint_writes=0, formal_training_started=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('passed','engine','counts','action_counts','max_float_difference','formal_training_started')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
