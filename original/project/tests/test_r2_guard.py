import unittest

import numpy as np

from tools import r2_guard as g


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.group = np.repeat(np.arange(8), 60)
        self.label = self.rng.random(480) < .4

    def test_auc_handles_ties_and_single_class(self):
        self.assertEqual(g.auc([0, 0, 1, 1], [False, False, True, True]), 1.)
        self.assertEqual(g.auc([1, 1, 1, 1], [False, True, False, True]), .5)
        self.assertEqual(g.auc([0, 1, 0, 1], [True, False, True, False]), 0.)
        self.assertIsNone(g.auc([0, 1], [True, True]))

    def test_gate_flags_a_signal_that_copies_the_label(self):
        r2 = np.where(self.label, 1., -1.)
        result = g.leak_gate(r2, self.label, self.group, range(4, 8), seed=1, shuffles=200)
        self.assertEqual(result['auc_real'], 1.)
        self.assertAlmostEqual(result['auc_shuffled'], .5, delta=.03)
        self.assertTrue(result['leaked'])

    def test_gate_finds_a_reversed_relation_too(self):
        result = g.leak_gate(np.where(self.label, -1., 1.), self.label, self.group, range(4, 8), seed=1, shuffles=200)
        self.assertEqual(result['auc_real'], 1.)

    def test_gate_passes_an_unrelated_signal(self):
        r2 = self.rng.choice([-1., 1.], 480)
        result = g.leak_gate(r2, self.label, self.group, range(4, 8), seed=1, shuffles=200)
        self.assertFalse(result['leaked'])
        self.assertLess(abs(result['margin']), g.LEAK_MARGIN)

    def test_gate_ignores_a_margin_that_shuffling_reaches_by_chance(self):
        label = np.array([True, False] * 4)
        group = np.repeat(np.arange(2), 4)
        r2 = np.array([1., -1., 1., 1.] * 2)
        result = g.leak_gate(r2, label, group, (1,), seed=1, shuffles=400)
        self.assertGreater(result['margin'], g.LEAK_MARGIN)
        self.assertGreater(result['p'], g.LEAK_P_MAX)
        self.assertFalse(result['leaked'])

    def test_gate_reports_a_split_without_both_classes(self):
        result = g.leak_gate(np.ones(480), np.zeros(480, bool), self.group, range(4, 8), seed=1, shuffles=5)
        self.assertIsNone(result['leaked'])

    def test_partial_correlation_removes_what_the_input_explains(self):
        x = self.rng.normal(size=480)
        through_input = g.partial_correlation(2 * x, 3 * x + self.rng.normal(size=480), x[:, None], seed=1, permutations=300)
        self.assertIsNone(through_input['leaked'])
        extra = self.rng.normal(size=480)
        direct = g.partial_correlation(x + extra, x + extra + .1 * self.rng.normal(size=480), x[:, None], seed=1, permutations=300)
        self.assertTrue(direct['leaked'])
        unrelated = g.partial_correlation(self.rng.normal(size=480), self.rng.normal(size=480), x[:, None], seed=1, permutations=300)
        self.assertFalse(unrelated['leaked'])

    def test_shares_of_a_unit_sign_next_to_a_small_reward(self):
        r1 = .01 + .001 * self.rng.normal(size=480)
        s = g.shares(r1, self.rng.choice([-1., 1.], 480))
        self.assertGreater(s['magnitude'], 90)
        self.assertGreater(s['variance'], .999)
        self.assertEqual(s['larger_than_r1'], 480)
        self.assertGreater(s['sign_flips'], 200)
        self.assertEqual(g.verdict(s)[0], 'redline')

    def test_verdict_levels_and_flags(self):
        base = dict(magnitude=.05, variance=.05, rho=.3)
        self.assertEqual(g.verdict(base), ('ok', []))
        self.assertEqual(g.verdict(base | dict(variance=.15, rho=.01)), ('above_target', ['noise']))
        self.assertEqual(g.verdict(base | dict(magnitude=.25, rho=.9)), ('redline', ['redundant']))

    def test_scale_follows_the_median(self):
        first = g.track_scale(None, [.01, -.02, 5.])
        self.assertEqual(first, .02)
        self.assertAlmostEqual(g.track_scale(first, [.01] * 3), g.EMA_DECAY * .02 + (1 - g.EMA_DECAY) * .01)

    def test_shape_is_centered_capped_and_follows_the_scale(self):
        r1 = np.array([.01, .01, .01, .01, 0., 0.])
        signal = np.array([3., -3., 3., -3., 3., -3.])
        out = g.shape(r1, signal, s1=.01, beta=.05)
        self.assertAlmostEqual(out.sum(), 0., places=12)
        self.assertAlmostEqual(out[0], .05 * .01 * np.tanh(3.))
        floor = g.DELTA_FRAC * .01
        self.assertAlmostEqual(out[4], floor)
        self.assertAlmostEqual(out[5], -floor)
        halved = g.shape(r1 / 2, signal, s1=.005, beta=.05)
        np.testing.assert_allclose(halved, out / 2)
        self.assertTrue(np.all(np.abs(out) <= g.KAPPA * np.abs(r1) + floor + 1e-15))

    def test_shape_leaves_ineligible_decisions_alone_and_bounds_beta(self):
        out = g.shape([.01, .01, .01], [1., 1., -1.], .01, .05, eligible=[True, False, True])
        self.assertEqual(out[1], 0.)
        self.assertAlmostEqual(out[0], -out[2])
        self.assertTrue(np.all(g.shape([.01, .02], [1., -1.], .01, 0.) == 0))
        with self.assertRaises(ValueError):
            g.shape([.01], [1.], .01, g.BETA_MAX + .01)
        with self.assertRaises(ValueError):
            g.shape([.01], [1.], 0., .05)


if __name__ == '__main__':
    unittest.main()
