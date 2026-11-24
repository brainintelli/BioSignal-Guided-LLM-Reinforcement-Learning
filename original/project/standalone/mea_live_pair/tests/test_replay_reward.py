import unittest

from replay_reward import ReplayRewardMix, emergency_drink_label


class ReplayRewardTests(unittest.TestCase):
    def test_low_water_requires_executed_drink_and_net_replenishment(self):
        self.assertEqual(emergency_drink_label('drink',20,69.92,20),1)
        self.assertEqual(emergency_drink_label('drink',0,49.92,20),1)
        self.assertEqual(emergency_drink_label('drink',20.01,69.93,20),0)
        self.assertEqual(emergency_drink_label('drink',1.6,1.6,20),0)
        self.assertEqual(emergency_drink_label('drink',1.6,1.52,20),0)
        self.assertEqual(emergency_drink_label('move',1.6,51.52,20),0)

    def setUp(self):
        self.mix = ReplayRewardMix(.009775, .1, 1., .5, .99985)

    def test_opposing_sign_cannot_reverse_small_native_reward(self):
        for native in (.00001, -.00001):
            sign = -1 if native > 0 else 1
            scaled, contribution, total = self.mix.score(native, sign, 4)
            self.assertAlmostEqual(scaled, sign*.1*abs(native))
            self.assertAlmostEqual(contribution, sign*.05*abs(native))
            self.assertGreater(total*native, 0)
        self.assertEqual(self.mix.score(0., -1, 4), (0., 0., 0.))

    def test_magnitude_capped_by_calibration_and_duration_once(self):
        ticks = 4
        expected = -.1*.009775*sum(.99985**i for i in range(ticks))
        scaled, part, total = self.mix.score(1., -1, ticks)
        self.assertAlmostEqual(scaled, expected)
        self.assertAlmostEqual(part, .5*expected)
        self.assertAlmostEqual(total, 1.+part)
        self.assertEqual(self.mix.score(-.03, 0, 2), (0., 0., -.03))

    def test_control_logs_calibrated_feedback_with_zero_contribution(self):
        spec = dict(alpha=1, beta=.5, reward_mix=dict(schema='bounded_native_r1.v1',
                    reference_per_tick=.01, scale_fraction=.1, gamma_tick=1.))
        control = ReplayRewardMix.from_study(spec, 1., False)
        scaled, part, total = control.score(-.02, 1, 2)
        self.assertAlmostEqual(scaled, .002)
        self.assertEqual((part,total), (0.,-.02))
        treatment = ReplayRewardMix.from_study(spec, 1., True)
        self.assertAlmostEqual(treatment.score(-.02, 1, 2)[1], .001)
        with self.assertRaises(ValueError):
            ReplayRewardMix.from_study(spec, .5, True)

    def test_invalid_ratio_or_label(self):
        with self.assertRaises(ValueError):
            ReplayRewardMix(.01, .1, 1., 10., .99985)
        with self.assertRaises(ValueError):
            self.mix.score(.01, 2, 1)
        with self.assertRaises(ValueError):
            self.mix.score(float('nan'), 1, 1)
        with self.assertRaises(ValueError):
            self.mix.score(.01, 1, 0)
