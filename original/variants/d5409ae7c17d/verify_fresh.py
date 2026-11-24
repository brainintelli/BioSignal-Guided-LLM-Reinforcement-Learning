"""Replay all short-probe actions in Godot, without loading any model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.m8x_env import GodotEnv
from agent.m8x_contract import SurvivalReward, encode, target_messages
from tools.audit_m6_run import assert_equivalent

parser = argparse.ArgumentParser()
parser.add_argument('--godot', required=True)
parser.add_argument('--evidence', default=str(Path(__file__).resolve().parent))
args = parser.parse_args()
folder = Path(args.evidence)
comparison = json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
assert comparison['passed'] and all(comparison['gates'].values())
assert not comparison['formal_training_started'] and comparison['checkpoints_saved'] == 0
for name,digest in comparison['manifest']['source_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
registration = json.loads((ROOT/'output/rl_report_20261107/registration.json').read_text(encoding='utf-8'))
for name,digest in registration['source_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
baseline = json.loads((folder/'baseline_probabilities.json').read_text())
exact = json.loads((folder/'exact_probabilities.json').read_text())
assert baseline == exact
for mode in ('baseline','exact','compact','prefix'):
    probabilities = json.loads((folder/(mode+'_probabilities.json')).read_text())
    for model in ('actor','reference'):
        assert len(probabilities[model]) == 64
        assert all(abs(sum(row)-1)<1e-5 and all(0<=x<=1 for x in row) for row in probabilities[model])
counts = {}
for mode in ('baseline', 'compact', 'prefix'):
    rows = [json.loads(x) for x in (folder / (mode+'_fresh_transitions.jsonl')).read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 192
    checked = 0
    for slot in range(4):
        selected = [r for r in rows if r['env'] == slot]
        with GodotEnv(args.godot, str(ROOT)) as env:
            state = env.reset(selected[0]['seed'], 'open3')
            tracker = SurvivalReward(state['observation'])
            for row in selected:
                assert row['source'] == 'model' and row['mode'] == mode
                assert_equivalent(state['observation'], row['observation'], 'observation')
                encoded = encode(state['observation'])
                v,t = row['verb_index'],row['target_index']
                assert_equivalent(encoded['messages'], row['model_input'], 'model_input')
                assert_equivalent(encoded['branches'][v][t], row['action'], 'legal_action')
                target = target_messages(encoded,v) if len(encoded['branches'][v])>1 else None
                assert_equivalent(target, row['target_input'], 'target_input')
                nxt = env.step(row['action'])
                assert_equivalent(nxt['transition'], row['transition'], 'transition')
                assert_equivalent(nxt['observation'], row['next_observation'], 'next_observation')
                assert_equivalent(tracker.score(nxt['transition'],nxt['observation']), row['reward'], 'reward')
                assert nxt['terminated'] == row['terminal']
                state = nxt
                checked += 1
    counts[mode] = checked
result = dict(passed=True, real_godot_replayed_steps=sum(counts.values()), arms=counts,
              rewards_recomputed=sum(counts.values()), inputs_and_legal_actions_verified=True,
              frozen_core_files_verified=len(registration['source_hashes']),
              benchmark_source_hashes_verified=True, exact_probabilities_equal=True,
              scope='short diagnostic trajectories; no claim about long-run convergence')
(folder/'replay_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
