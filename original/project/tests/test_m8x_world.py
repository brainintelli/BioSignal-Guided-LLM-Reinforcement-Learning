"""Persistent real Godot bridge for the separately versioned candidate world."""
import hashlib
import importlib
import os
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
GODOT=Path(os.environ.get('MEADOW_GODOT',str(ROOT/'.tools/godot/Godot_v4.7.2-stable_win64.exe')))


class ExplorationWorldTests(unittest.TestCase):
    def test_real_rules_identity_reward_and_terminal(self):
        Env=importlib.import_module('agent.m8x_env').GodotEnv
        c=importlib.import_module('agent.m8x_contract')
        expected=hashlib.sha256((ROOT/'configs/m8_exploration_v1.json').read_bytes()).hexdigest()
        with Env(str(GODOT),str(ROOT)) as env:
            first=env.reset(2304);obs=first['observation'];self.assertEqual(obs['profile'],'open3')
            self.assertEqual(obs['world_version'],5);self.assertEqual(obs['balance_sha256'],expected)
            tracker=c.SurvivalReward(obs)
            step=env.step({'verb':'wait'});r=tracker.score(step['transition'],step['observation'])
            self.assertAlmostEqual(step['transition']['after']['hunger'],81.8)
            self.assertAlmostEqual(step['transition']['after']['thirst'],77.92)
            self.assertEqual(step['transition']['world_version'],5);self.assertEqual(r['elapsed_ticks'],1)
            step=env.step({'verb':'rest'});tracker.score(step['transition'],step['observation'])
            self.assertEqual(step['transition']['elapsed_ticks'],3)
            self.assertAlmostEqual(step['transition']['after']['thirst'],77.68)
            with self.assertRaises(RuntimeError):env.step({'verb':'teleport'})
            self.assertEqual(env.call('observe')['observation']['tick'],4)
            while not step['terminated'] and step['observation']['tick']<1800:
                step=env.step({'verb':'wait'});tracker.score(step['transition'],step['observation'])
            self.assertTrue(step['terminated']);self.assertEqual(step['observation']['tick'],486)
            with self.assertRaises(RuntimeError):env.step({'verb':'wait'})
            with self.assertRaises(RuntimeError):env.reset(2304,'open2')


if __name__=='__main__':unittest.main()
