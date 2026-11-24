"""Representation checks against recorded observations; no model or training."""
import copy
import json
from pathlib import Path
import unittest

import semantic_candidate as candidate

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / 'output/rl_kv2_20261107/optimized_transitions.jsonl'


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in TRACE.read_text(encoding='utf-8').splitlines()]

    def api(self, name):
        self.assertTrue(callable(getattr(candidate, name, None)), 'Missing candidate API: ' + name)
        return getattr(candidate, name)

    def test_time_boundaries_and_lossless_cycle(self):
        encode, decode = self.api('time_state'), self.api('time_tick')
        self.assertEqual(encode(139), {'时段': '黄昏', '距离入夜还剩刻数': 1})
        self.assertEqual(encode(140), {'时段': '夜间', '距离天亮还剩刻数': 40})
        self.assertEqual(encode(179), {'时段': '夜间', '距离天亮还剩刻数': 1})
        for tick in range(720):
            self.assertEqual(decode(encode(tick)), tick % 180)

    def test_map_roundtrip_with_unknown_and_blocked_cells(self):
        encode, decode = self.api('map_state'), self.api('map_rows')
        rows = ['?.s#~', 'fffff', '*****', '.....', '~~~~~']
        for mode in ('grid', 'sparse'):
            self.assertEqual(decode(encode(rows, mode)), rows)

    def test_visit_counts_roundtrip_including_zero(self):
        encode, decode = self.api('visit_state'), self.api('visit_rows')
        rows = [[0, 1, 2, 3, 4], [0]*5, [0, 0, 98, 0, 0], [0]*5, [1]*5]
        self.assertEqual(decode(encode(rows)), rows)

    def test_real_actions_are_semantic_bijective_and_one_tile(self):
        build = self.api('build')
        for row in self.rows:
            result = build(row)
            actions = result['action_catalog']
            actual = {json.dumps(x['engine_action'], sort_keys=True) for x in actions}
            expected = {json.dumps(x, sort_keys=True) for x in row['observation']['allowed_actions']}
            self.assertEqual(actual, expected)
            self.assertEqual(len({x['text'] for x in actions}), len(actions))
            for item in actions:
                self.assertFalse(item['text'].isascii())
                if item['engine_action']['verb'] == 'move':
                    self.assertIn(item['target'], ('向上', '向右', '向下', '向左'))
                    self.assertTrue(item['text'].endswith('走1格'))

    def test_known_action_target_has_correct_resource_and_position(self):
        result = self.api('build')(self.rows[0])
        action = next(x for x in result['action_catalog'] if x['engine_action'] == self.rows[0]['action'])
        self.assertEqual(action['text'], '前往树木（左3格、上1格），走1格')
        self.assertEqual(action['engine_action'], {'verb':'navigate','target_id':'r_44_46'})
        self.assertIn({'名称':'饮水岸边（右1格）','已知路程':1,'可用':True,
                       '再生还需刻数':0,'可执行':['前往']}, result['state']['地标'])

    def test_all_real_facts_retained_except_nonsemantic_aliases(self):
        build = self.api('build')
        decode_map, decode_visits = self.api('map_rows'), self.api('visit_rows')
        for row in self.rows:
            facts = json.loads(row['model_input'][1]['content'])
            for mode in ('grid', 'sparse'):
                state = build(row, mode)['state']
                self.assertEqual(state['身体'], facts['剩余数值'])
                self.assertEqual(state['背包'], facts['背包'])
                self.assertEqual(decode_map(state['地形']), facts['局部5x5'])
                self.assertEqual(decode_visits(state['走过的位置']), facts['局部踏入次数5x5'])
                self.assertEqual(len(state['地标']), len(facts['地标[别名,种类,dx,dy,路程,可用,再生步]']))
                for actual, old in zip(state['地标'], facts['地标[别名,种类,dx,dy,路程,可用,再生步]']):
                    self.assertEqual(actual['已知路程'], old[4])
                    self.assertEqual(actual['可用'], bool(old[5]))
                    self.assertEqual(actual['再生还需刻数'], old[6])
                self.assertEqual(len(state['最近动作']), len(facts['过去[动作,dx,dy,目标种类相对位置,变化]']))
                self.assertNotIn('环境', state)
                self.assertNotIn('日内步', state)

    def test_hidden_raw_fields_cannot_enter_actor_text(self):
        build = self.api('build')
        row = copy.deepcopy(self.rows[0])
        before = build(row)
        row['observation']['hidden_map'] = 'SECRET_MAP'
        row['observation']['recommended_action'] = 'GO_WEST_NOW'
        row['observation']['visited_count'] = 99999
        row['value'] = 1e9
        row['reward'] = {'reward': 1e9}
        self.assertEqual(before, build(row))

    def test_original_record_is_not_mutated(self):
        build = self.api('build')
        row = copy.deepcopy(self.rows[0])
        before = copy.deepcopy(row)
        build(row)
        self.assertEqual(row, before)

    def test_concise_keeps_landmark_semantics_and_time(self):
        render = self.api('render_concise')
        state = self.api('build')(self.rows[0])['state']
        text = render(state)
        for item in state['地标']:
            self.assertIn(item['名称'], text)
        self.assertIn('140刻', text)
        self.assertIn('未写再生表示0刻', text)
        self.assertNotIn('open3', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
