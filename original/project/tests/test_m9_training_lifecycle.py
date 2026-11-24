"""Real train-loop files and native deaths; scripted policy is not model evidence."""
from contextlib import ExitStack
import gzip
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GODOT = Path(os.environ.get('MEADOW_GODOT', str(ROOT / '.tools/godot/Godot_v4.7.2-stable_win64.exe')))


class RolloutsCollected(Exception):
    """Stop before the second optimizer call; no fake checkpoint is produced."""


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Node runtime supplies torch')
class TrainingLifecycle(unittest.TestCase):
    def test_natural_deaths_preserve_other_slots_resets_and_next_rollout(self):
        import torch
        import agent.m9_run as run
        from agent.m9_env import GodotEnv
        from agent.m9_contract import encode

        class ScriptedPolicy:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1)
                self.critic = torch.nn.Linear(1, 1)

            def digest(self):
                return 'scripted-io-test-not-a-trained-model'

            def prepare(self, observations):
                return [encode(obs) for obs in observations]

            def act(self, prepared, with_reference):
                # Deterministic legal wait triggers starvation in primed native worlds.
                return dict(actions=[{'verb': 'wait'} for _ in prepared],
                    verbs=[7] * len(prepared), targets=[0] * len(prepared),
                    values=[0.] * len(prepared), logp=[0.] * len(prepared),
                    verb_logp=[0.] * len(prepared), reference_verb=[[0.] * 7 + [1.]] * len(prepared),
                    reference_target=[[1.]] * len(prepared))

            def values(self, prepared):
                return torch.zeros(len(prepared))

        class PrimedNativeEnvironment:
            def __init__(self, config):
                self.native = GodotEnv(config.godot, config.project,
                    map_mode=config.map_mode, navigation_steps=config.navigation_steps)
                self.first_reset = True

            def reset(self, seed):
                state = self.native.reset(seed)
                if self.first_reset:
                    # Only this fixture advances real wait actions. Production reset is untouched.
                    for _ in range(600):
                        if state['observation']['player']['health'] <= 1.2:
                            break
                        state = self.native.step({'verb': 'wait'})
                    if state['terminated'] or not 0 < state['observation']['player']['health'] <= 1.2:
                        raise AssertionError('Native near-death fixture was not reached')
                    self.first_reset = False
                return state

            def step(self, action):
                return self.native.step(action)

            def close(self):
                self.native.close()

        for mode, navigation_steps in [('finite', 1), ('infinite_freeze', 4)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=ROOT / 'output/runtime_tests') as tmp:
                out = Path(tmp)
                config = SimpleNamespace(godot=str(GODOT), project=str(ROOT), map_mode=mode,
                    navigation_steps=navigation_steps, reward_variant='supply', gamma_tick=.9995,
                    gae_lambda=.999, evaluation_seeds=[3990000], actor_lr=2e-6, critic_lr=1e-4,
                    train_seed_start=3980000, train_seed_count=20, rng_seed=123,
                    envs=2, updates=2, rollout_steps=3)
                optimizer_calls = 0

                def stop_after_second_rollout(*args):
                    nonlocal optimizer_calls
                    optimizer_calls += 1
                    if optimizer_calls == 2:
                        raise RolloutsCollected()
                    return {'optimizer_steps': 0}

                # The expensive LM/PPO is outside this logging test; native worlds,
                # rewards, train control flow, resets, GAE and every file write are real.
                with ExitStack() as patches:
                    patches.enter_context(patch.object(run, 'environment', PrimedNativeEnvironment))
                    patches.enter_context(patch.object(run, 'evaluate', return_value={'summary': {}}))
                    patches.enter_context(patch('agent.m8_train.update', side_effect=stop_after_second_rollout))
                    patches.enter_context(patch('tools.benchmark_m8s3.selected_logps',
                        side_effect=lambda policy, rows, verbs, targets: [0.] * len(rows)))
                    patches.enter_context(patch('torch.cuda.synchronize'))
                    with self.assertRaises(RolloutsCollected):
                        run.train(ScriptedPolicy(), config, out)

                def records(path):
                    with gzip.open(path, 'rt', encoding='utf-8') as stream:
                        return [json.loads(line) for line in stream]

                first = records(out / 'rollout_0001.jsonl.gz')
                second = records(out / 'rollout_0002.jsonl.gz')
                episodes = [json.loads(line) for line in (out / 'episodes.jsonl').read_text().splitlines()]
                self.assertEqual(len(first), 6)
                self.assertEqual(len(second), 6)
                self.assertEqual([r['transition']['after']['dead'] for r in first], [True, True, False, False, False, False])
                self.assertEqual([r['seed'] for r in first], [3980000, 3980001, 3980002, 3980003, 3980002, 3980003])
                self.assertEqual([r['transition']['after']['tick'] for r in first[2:]], [1, 1, 2, 2])
                self.assertEqual([r['transition']['after']['tick'] for r in second], [3, 3, 4, 4, 5, 5])
                self.assertEqual([e['seed'] for e in episodes], [3980000, 3980001])
                self.assertTrue(all(e['final']['dead'] for e in episodes))
                self.assertTrue(all('observation' in r and 'transition' in r for r in first + second))
                self.assertTrue(all('transition' not in e and 'final' in e for e in episodes))
                self.assertEqual(json.loads((out / 'metrics.jsonl').read_text())['decisions'], 6)


if __name__ == '__main__':
    unittest.main()
