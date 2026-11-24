"""New-world input boundary checks: observations are facts, never reward labels."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from tests.test_m6_semantic_contract import observation as legacy_observation

ROOT = Path(__file__).resolve().parents[1]


def observation():
    obs = legacy_observation()
    obs['player'].pop('energy')
    obs.update(profile='open4', world_version=6, environment_version='m8_survival_v2',
               balance_sha256=hashlib.sha256((ROOT/'configs/m8_survival_v2.json').read_bytes()).hexdigest())
    return obs


class SurvivalInputTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('agent.m8s3_contract')

    def test_complete_legal_action_set_and_no_retired_state_fields(self):
        obs = observation()
        before = copy.deepcopy(obs)
        row = self.c.encode(obs)
        self.assertEqual(obs, before)
        self.assertCountEqual([a for branch in row['branches'] for a in branch], obs['allowed_actions'])
        serialized = json.dumps({'messages':row['messages'], 'state':row['semantic_state']}, ensure_ascii=False)
        for retired in ('体力', 'energy', '路程', '可前往', 'path_steps'):
            self.assertNotIn(retired, serialized)
        for branch in row['target_texts']:
            for target in branch:
                if target:
                    self.assertIn(target, '\n'.join(m['content'] for m in row['messages']))
        self.assertIn('地形', serialized)
        self.assertEqual(len(row['features']), 96, 'The removed reserve must not be a dummy critic feature')

    def test_reward_and_privileged_labels_cannot_change_actor_or_critic_inputs(self):
        obs = observation()
        clean = self.c.encode(obs)
        polluted = copy.deepcopy(obs)
        secret = 'PRIVATE_REWARD_ORACLE_SENTINEL_7391'
        for key in ('reward', 'return', 'advantage', 'value_target', 'future', 'best_action', 'recommended_action', 'task', 'hidden_map'):
            polluted[key] = secret
        polluted['player']['reward'] = secret
        for item in polluted['nearby']:
            item['reward'] = secret
            item['best_action'] = secret
            item['path_steps'] = 987654
        candidate = self.c.encode(polluted)
        for field in ('messages', 'semantic_state', 'features', 'branches', 'target_texts'):
            self.assertEqual(candidate[field], clean[field], field)
        self.assertNotIn(secret, json.dumps(candidate, ensure_ascii=False))

    def test_system_rules_are_static_and_only_describe_mechanics(self):
        first = observation()
        second = copy.deepcopy(first)
        second['player'].update(health=44., hunger=22., thirst=31.)
        second['tick'] = 142
        a, b = self.c.encode(first), self.c.encode(second)
        self.assertEqual(a['messages'][0], b['messages'][0])
        self.assertNotEqual(a['messages'][1:], b['messages'][1:])
        spatial = self.c.render_scene(a['semantic_state'])
        self.assertTrue(a['messages'][1]['content'].startswith(spatial))
        self.assertTrue(b['messages'][1]['content'].startswith(spatial))
        self.assertLess(a['messages'][1]['content'].index('地标'), a['messages'][1]['content'].index('身体'))
        rules = a['messages'][0]['content']
        for forbidden in ('reward', 'advantage', 'gamma', '0.9995', '势能', '奖励', '最优动作', '体力', '路程'):
            self.assertNotIn(forbidden, rules.lower())
        for mechanic in ('一格', '饱食', '水分', '生命', '夜间', '营地'):
            self.assertIn(mechanic, rules)

    def test_old_identity_and_wrong_rule_hash_fail_closed(self):
        obs = observation()
        for mutation in ({'profile':'open3'}, {'world_version':5}, {'world_version':True},
                         {'environment_version':'m8_exploration_v1'}, {'balance_sha256':'0'*64}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.c.encode({**obs, **mutation})

    def test_landmarks_do_not_replace_local_water_and_unknown_tiles(self):
        first = observation()
        second = copy.deepcopy(first)
        second['local_map']['rows'] = ['?....', '.~...', '.....', '.....', '.....']
        a, b = self.c.encode(first), self.c.encode(second)
        self.assertEqual(a['target_texts'], b['target_texts'])
        self.assertNotEqual(a['messages'], b['messages'])
        self.assertIn('水', b['messages'][1]['content'])
        self.assertTrue(any(word in b['messages'][1]['content'] for word in ('未知', '未观察')))

    def test_training_potential_has_exactly_three_survival_dimensions(self):
        state = dict(health=100., hunger=60., thirst=60., dead=False)
        self.assertEqual(self.c.potential(state), 1.)
        self.assertEqual(self.c.potential({**state, 'energy':0.}), self.c.potential({**state, 'energy':100.}))
        self.assertEqual(self.c.potential({**state, 'dead':True}), 0.)
        self.assertAlmostEqual(self.c.potential({**state, 'hunger':0.}), .75)
        self.assertAlmostEqual(self.c.potential({**state, 'thirst':0.}), .75)

    def test_reward_terminal_boundary_and_reject_incomplete_trace(self):
        initial = observation()
        initial['player'].update(health=.1, hunger=0., thirst=0.)
        after = copy.deepcopy(initial)
        after['tick'] = 1
        after['player']['health'] = 0.
        after['allowed_actions'] = []

        def snapshot(obs, dead):
            p = obs['player']
            return dict(tick=obs['tick'], dead=dead, health=p['health'], hunger=p['hunger'],
                        thirst=p['thirst'], player={'x':p['x'], 'y':p['y']})

        before_state, after_state = snapshot(initial, False), snapshot(after, True)
        transition = dict(before=before_state, after=after_state, elapsed_ticks=1, trace=[after_state])
        tracker = self.c.SurvivalReward(initial)
        with self.assertRaises(ValueError):
            tracker.score({**transition, 'trace':[]}, after)
        result = tracker.score(transition, after)
        self.assertEqual(result['components']['survival'], 0., 'Death tick is not rewarded as alive')
        self.assertAlmostEqual(result['components']['potential'], -self.c.potential(before_state))
        with self.assertRaises(ValueError):
            tracker.score(transition, after)


if __name__ == '__main__':
    unittest.main()
