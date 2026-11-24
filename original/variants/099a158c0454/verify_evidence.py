"""Replay every fresh diagnostic action in real local Godot and recompute rewards."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.m8x_env import GodotEnv
from agent.m8x_contract import SurvivalReward, encode, target_messages, critic_features
from tools.audit_m6_run import assert_equivalent

folder = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (folder/'fresh_transitions.jsonl').read_text(encoding='utf-8').splitlines()]
profile = json.loads((folder/'profile.json').read_text(encoding='utf-8'))
registration = json.loads((folder/'registration.json').read_text(encoding='utf-8'))
assert len(rows) == 288
assert sum(r['measured'] for r in rows) == 256
assert profile['optimizer_steps'] == 0 and not profile['checkpoint_saved']
assert profile['actor_digest_before'] == profile['actor_digest_after']
assert profile['reference_digest_before'] == profile['reference_digest_after']
assert profile['probe_sha256'] == hashlib.sha256((folder/'profile_probe.py').read_bytes()).hexdigest()
for rel, digest in registration['source_hashes'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
checked = 0
for slot in range(4):
    selected = [r for r in rows if r['env'] == slot]
    with GodotEnv(str(ROOT/'.tools/godot/Godot_v4.7.2-stable_win64_console.exe'), str(ROOT)) as env:
        state = env.reset(selected[0]['seed'], 'open3')
        tracker = SurvivalReward(state['observation'])
        for row in selected:
            assert_equivalent(state['observation'], row['observation'], 'observation')
            encoded = encode(state['observation'])
            v,t = row['verb_index'],row['target_index']
            assert_equivalent(encoded['messages'], row['model_input'], 'model_input')
            assert_equivalent(encoded['facts'], row['facts'], 'facts')
            assert_equivalent(critic_features(encoded), row['critic_features'], 'critic_features')
            assert_equivalent(encoded['branches'][v][t], row['action'], 'action')
            assert_equivalent(target_messages(encoded,v) if len(encoded['branches'][v])>1 else None, row['target_input'], 'target_input')
            nxt = env.step(row['action'])
            assert_equivalent(nxt['transition'], row['transition'], 'transition')
            assert_equivalent(nxt['observation'], row['next_observation'], 'next_observation')
            assert_equivalent(tracker.score(nxt['transition'],nxt['observation']),row['reward'],'reward')
            assert nxt['terminated'] == row['terminal']
            checked += 1
            state = nxt
result = dict(passed=True, real_godot_replayed_steps=checked, worlds=4,
              rewards_recomputed=checked, inputs_and_actions_checked=checked,
              local_frozen_source_files_checked=len(registration['source_hashes']),
              probe_script_digest_verified=True, weight_digests_unchanged=True,
              scope='fresh diagnostic only; no historical full-run or CUDA RNG acceptance audit')
(folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
