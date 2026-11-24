"""End-to-end real Godot observations, state encoding and reward boundaries."""
import importlib
import math
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
GODOT = Path(os.environ.get('MEADOW_GODOT', str(ROOT/'.tools/godot/Godot_v4.7.2-stable_win64.exe')))


def assert_no_energy(test, value):
    if isinstance(value, dict):
        test.assertNotIn('energy', value)
        for item in value.values():
            assert_no_energy(test, item)
    elif isinstance(value, list):
        for item in value:
            assert_no_energy(test, item)


class SurvivalWorldTests(unittest.TestCase):
    def test_real_protocol_rewards_and_both_single_tile_branches(self):
        Env = importlib.import_module('agent.m8s3_env').GodotEnv
        c = importlib.import_module('agent.m8s3_contract')
        with Env(str(GODOT), str(ROOT)) as env:
            obs = env.reset(2304)['observation']
            self.assertEqual((obs['profile'], obs['world_version']), ('open4', 6))
            tracker = c.SurvivalReward(obs)
            row = c.encode(obs)
            for verb in ('move', 'navigate', 'rest', 'wait'):
                action = next(a for a in obs['allowed_actions'] if a['verb'] == verb)
                result = env.step(action)
                assert_no_energy(self, result)
                reward = tracker.score(result['transition'], result['observation'])
                self.assertTrue(math.isfinite(reward['reward']))
                self.assertNotIn('reward', result['observation'])
                if verb in ('move', 'navigate'):
                    before, after = (result['transition'][k]['player'] for k in ('before', 'after'))
                    self.assertEqual(abs(after['x']-before['x'])+abs(after['y']-before['y']), 1)
                    self.assertEqual(result['transition']['elapsed_ticks'], 1)
                obs = result['observation']
                row = c.encode(obs)
                self.assertEqual(len(row['features']), 96)
            with self.assertRaises(RuntimeError):
                env.reset(2304, 'open3')


if __name__ == '__main__':
    unittest.main()
