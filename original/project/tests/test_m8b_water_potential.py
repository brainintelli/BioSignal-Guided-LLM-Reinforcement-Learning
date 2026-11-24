"""Candidate shaping invariants; these tests are not model-quality evidence."""
import copy
import importlib
import math
import unittest


def facts(thirst=20, distance=16):
    landmarks = [] if distance is None else [["lake", "water", 7, -4, distance, True, 0]]
    return {"剩余数值": {"生命": 80, "饱食": 70, "水分": thirst, "体力": 60},
            "地标[别名,种类,dx,dy,路程,可用,再生步]": landmarks}


class WaterPotentialTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("agent.m8b_water_potential")

    def phi(self, state, dead=False, **overrides):
        config = dict(weight=1., urgency_threshold=60., distance_scale=32.)
        config.update(overrides)
        return self.module.water_potential(state, dead=dead, **config)

    def test_approaching_known_water_increases_potential_when_thirsty(self):
        self.assertGreater(self.phi(facts(distance=8)), self.phi(facts(distance=16)))

    def test_full_water_has_no_direction_preference(self):
        self.assertEqual(self.phi(facts(80, 0)), self.phi(facts(80, None)))

    def test_drinking_at_water_does_not_remove_the_potential(self):
        self.assertEqual(self.phi(facts(0, 0)), self.phi(facts(100, 0)))

    def test_unknown_and_unreachable_water_use_bounded_distance(self):
        self.assertEqual(self.phi(facts(0, None)), 0.)
        self.assertEqual(self.phi(facts(0, -1)), 0.)
        self.assertEqual(self.phi(facts(0, 999)), 0.)

    def test_only_nearest_reachable_observed_water_matters(self):
        state = facts(20, 20)
        state["地标[别名,种类,dx,dy,路程,可用,再生步]"].extend([
            ["blocked", "water", 0, 0, -1, True, 0],
            ["food", "berry", 0, 0, 0, True, 0],
            ["near", "water", 100, 100, 8, True, 0]])
        self.assertEqual(self.phi(state), self.phi(facts(20, 8)))

    def test_no_hidden_map_or_action_input_and_no_mutation(self):
        state = facts()
        original = copy.deepcopy(state)
        baseline = self.phi(state)
        state["hidden_map"] = {"water": [0, 0]}
        state["recommended_action"] = "drink"
        self.assertEqual(self.phi(state), baseline)
        del state["hidden_map"], state["recommended_action"]
        self.assertEqual(state, original)

    def test_terminal_is_exactly_zero(self):
        self.assertEqual(self.phi({}, dead=True), 0.)

    def test_nonnegative_and_bounded_for_valid_states(self):
        for thirst in (0, 20, 60, 100):
            for distance in (None, -1, 0, 10, 32, 1000):
                value = self.phi(facts(thirst, distance), weight=.75)
                self.assertGreaterEqual(value, 0.)
                self.assertLessEqual(value, .75)

    def test_zero_weight_disables_candidate(self):
        self.assertEqual(self.phi(facts(), weight=0.), 0.)

    def test_bad_configuration_and_vitals_are_rejected(self):
        for change in ({"weight": -1}, {"weight": True}, {"weight": math.nan},
                       {"distance_scale": 0}, {"urgency_threshold": 101},
                       {"urgency_threshold": 0}, {"distance_scale": math.inf}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.phi(facts(), **change)
        for thirst in (-1, 101, True, math.nan, "20"):
            with self.subTest(thirst=thirst), self.assertRaises(ValueError):
                self.phi(facts(thirst))
        with self.assertRaises(ValueError):
            self.phi(facts(), dead=1)

    def test_bad_water_distances_are_rejected(self):
        for distance in (-2, True, math.inf, math.nan, "8"):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.phi(facts(distance=distance))

    def test_variable_duration_path_telescopes_at_natural_death(self):
        gamma = .9995
        potentials = [self.phi(facts(20, d)) for d in (16, 8, 30, 2)] + [0.]
        durations = [1, 3, 2, 4]
        total = 0.
        elapsed = 0
        for before, after, dt in zip(potentials, potentials[1:], durations):
            total += gamma**elapsed * self.module.shaping_reward(before, after, dt, gamma_tick=gamma)
            elapsed += dt
        self.assertAlmostEqual(total, -potentials[0], places=12)

    def test_closed_loop_has_no_positive_discounted_bonus(self):
        gamma = .9995
        potentials = [.8, .2, 1., .8]
        durations = [2, 1, 3]
        total = 0.
        elapsed = 0
        for before, after, dt in zip(potentials, potentials[1:], durations):
            total += gamma**elapsed * self.module.shaping_reward(before, after, dt, gamma_tick=gamma)
            elapsed += dt
        self.assertAlmostEqual(total, (gamma**elapsed-1)*potentials[0], places=12)
        self.assertLessEqual(total, 0.)

    def test_bad_discount_duration_or_potentials_are_rejected(self):
        for dt in (0, -1, True, 1.5):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.module.shaping_reward(.5, .6, dt, gamma_tick=.9995)
        for gamma in (0, 1.1, True, math.nan):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                self.module.shaping_reward(.5, .6, 1, gamma_tick=gamma)
        for value in (-.1, math.nan, math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.shaping_reward(value, .6, 1, gamma_tick=.9995)


if __name__ == "__main__":
    unittest.main()
