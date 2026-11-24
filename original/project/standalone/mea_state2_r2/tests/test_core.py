import importlib
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'standalone/mea_state2_r2'), str(ROOT / 'standalone/mea_binary_r2'), str(ROOT)]


class FeatureTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('mea_state2_r2.features')
        except ModuleNotFoundError:
            self.fail('Versioned state2 feature implementation is missing')

    def test_uniform_and_concentrated_activity_have_distinct_shapes(self):
        f = self.module()
        events = [(i * 2000, c, 1.) for i in range(10) for c in (1, 2)]
        x = f.extract(events, 0, 20000, (1, 2), 20000)
        self.assertAlmostEqual(x[0], math.log1p(10))
        self.assertAlmostEqual(x[1], 1.)
        self.assertAlmostEqual(x[2], 0.)
        self.assertAlmostEqual(x[3], 1.)
        clustered = f.extract([(i, 1, 1.) for i in range(20)], 0, 20000, (1, 2), 20000)
        self.assertAlmostEqual(clustered[0], x[0])
        self.assertEqual(clustered[1:], (.5, 1., 0.))

    def test_silence_is_valid_but_missing_and_bad_coordinates_are_not(self):
        f = self.module()
        self.assertEqual(f.extract([], 0, 20000, (1, 2), 20000), (0., 0., 0., 0.))
        for events in (None, [(20000, 1, 1.)], [(-1, 1, 1.)], [(0, 9, 1.)], [(0, 1, float('nan'))]):
            with self.assertRaises(ValueError):
                f.extract(events, 0, 20000, (1, 2), 20000)
        with self.assertRaises(ValueError):
            f.extract([], 0, 10000, (1, 2), 20000)

    def test_baseline_normalization_is_frozen_floored_clipped_and_rounded(self):
        f = self.module()
        norm = f.fit([(1., .5, .1, .5)] * 5)
        self.assertEqual(norm['scales'], [.1, .05, .05, .05])
        self.assertEqual(f.transform((1., .5, .1, .5), norm), (0., 0., 0., 0.))
        self.assertEqual(f.transform((100., 1., 1., 1.), norm), (1., 1., 1., 1.))
        self.assertEqual(f.transform((1.01, .5, .1, .5), norm)[0], .033)
        self.assertEqual(norm['centers'], [1., .5, .1, .5])
        wrong = dict(norm, schema='mea.features.p0.v1')
        with self.assertRaises(ValueError):
            f.transform((1., .5, .1, .5), wrong)


class ProtocolTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('mea_state2_r2.protocol')
        except ModuleNotFoundError:
            self.fail('Frozen experiment protocol implementation is missing')

    def test_shuffle_preserves_windows_and_split_without_fixed_points(self):
        p = self.module()
        for split, start, end in [('train', 0, 384), ('eval', 445, 595)]:
            order = p.assignment(0, 'D', split)
            self.assertEqual(sorted(order[:-1]), list(range(start, end)))
            self.assertTrue(all(i != j for i, j in zip(range(start, end), order[:-1])))
            self.assertEqual(order[-1], end)
            self.assertEqual(p.assignment(0, 'C', split), list(range(start, end + 1)))
        with self.assertRaises(ValueError):
            p.assignment(8, 'D', 'train')

    def test_reward_sign_duration_hysteresis_and_terminal_reset(self):
        p = self.module()
        engine = p.Rewards(True)
        def score(h, f, w, duration=1, terminal=False):
            return engine.score(.01, duration, terminal, {'H': h, 'F': f, 'W': w}, 'ctx')
        self.assertEqual(score(100, 100, 100)['r2_raw'], 1)
        neg = score(100, 100, 80, 4)
        self.assertEqual(neg['r2_raw'], -1)
        self.assertAlmostEqual(neg['r2_contribution'], -.001 * sum(.99985 ** k for k in range(4)))
        self.assertEqual(score(100, 100, 87)['r2_raw'], -1)
        dead = score(0, 0, 0, terminal=True)
        self.assertEqual(dead['r2_contribution'], 0)
        self.assertEqual(dead['total_reward'], .01)
        self.assertEqual(score(100, 100, 87)['r2_raw'], 1)
        self.assertEqual(neg['r2_source'], 'simulation_from_game_state')
        self.assertFalse(neg['real_output_verified'])

    def test_r1_only_arm_keeps_original_reward(self):
        p = self.module()
        r = p.Rewards(False).score(-.25, 4, False, {'H': 80, 'F': 70, 'W': 60}, 'x')
        self.assertEqual(r['total_reward'], -.25)
        self.assertEqual(r['r2_contribution'], 0.)
        self.assertFalse(r['r2_enabled'])


if __name__ == '__main__':
    unittest.main()
