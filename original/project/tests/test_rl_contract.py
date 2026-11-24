import copy
import unittest
from agent.rl_contract import RewardConfig, RewardTracker, encode_observation, gae, potential, transition_at_horizon


def state(tick=0, x=0, **kw):
    return dict(tick=tick, player={"x":x,"y":0}, health=100., hunger=80., thirst=80., energy=80.,
                damage=0., dead=False, **kw)


class ContractTests(unittest.TestCase):
    def observation(self):
        return {"tick":3,"profile":"open1","player":{**state()["player"],"health":90,"hunger":40,"thirst":50,"energy":70},
                "nearby":[{"id":"r_2_0","kind":"tree","x":2,"y":0,"available":True,"path_steps":1}],
                "allowed_actions":[{"verb":"navigate","target_id":"r_2_0"},{"verb":"move","target_id":"tile_1_0"},{"verb":"rest"}],
                "task":{"goal":"SECRET_MISSION_DO_NOT_INCLUDE"},"recommended_action":"SECRET_RECOMMENDATION"}

    def test_no_goal_and_lossless_all_candidates(self):
        obs = self.observation()
        before = copy.deepcopy(obs)
        encoded = encode_observation(obs)
        text = str(encoded["messages"])
        self.assertNotIn("SECRET", text)
        self.assertNotIn("r_2_0", text)
        self.assertCountEqual(encoded["actions"], obs["allowed_actions"])
        self.assertEqual(len(set(encoded["labels"])), len(obs["allowed_actions"]))
        self.assertEqual(obs, before)

    def test_input_key_and_candidate_order_do_not_change_policy_input(self):
        obs = self.observation()
        flipped = copy.deepcopy(obs)
        flipped["allowed_actions"] = [dict(reversed(list(a.items()))) for a in reversed(obs["allowed_actions"])]
        self.assertEqual(encode_observation(obs), encode_observation(flipped))

    def test_no_candidate_dropped_at_maximum(self):
        obs = self.observation()
        obs["allowed_actions"] = [{"verb":"move","target_id":f"tile_{x}_0"} for x in range(64)]
        self.assertEqual(len(encode_observation(obs)["actions"]), 64)
        obs["allowed_actions"].append({"verb":"wait"})
        with self.assertRaises(ValueError):
            encode_observation(obs)

    def test_reward_trace_is_required(self):
        with self.assertRaises(ValueError):
            RewardTracker(state()).score({"before":state(),"after":state(3),"elapsed_ticks":3,"trace":[state(3)]})

    def test_same_trajectory_discounted_segment_additivity(self):
        a,b,c,d = state(),state(1,x=1),state(2,x=2),state(3,x=3)
        b["thirst"] = 79.; c["thirst"] = 78.; d["thirst"] = 77.
        one = RewardTracker(a).score({"before":a,"after":d,"elapsed_ticks":3,"trace":[b,c,d]})
        tracker = RewardTracker(a)
        first = tracker.score({"before":a,"after":b,"elapsed_ticks":1,"trace":[b]})
        rest = tracker.score({"before":b,"after":d,"elapsed_ticks":2,"trace":[c,d]})
        for key in one["components"]:
            self.assertAlmostEqual(one["components"][key],first["components"][key]+.997*rest["components"][key], places=10)

    def test_novelty_revisit_and_cap(self):
        tracker = RewardTracker(state())
        previous = state()
        novelty = 0.
        for i in range(1, 101):
            next_state = state(i,x=i)
            reward = tracker.score({"before":previous,"after":next_state,"elapsed_ticks":1,"trace":[next_state]})
            novelty += reward["components"]["novelty"]
            previous = next_state
        self.assertAlmostEqual(novelty, 1.)
        back = state(101)
        self.assertEqual(tracker.score({"before":previous,"after":back,"elapsed_ticks":1,"trace":[back]})["components"]["novelty"],0.)

    def test_real_damage_not_hidden_by_healing(self):
        before,after = state(),state(1)
        before["health"] = 50.; after["health"] = 53.7; after["damage"] = .3
        reward = RewardTracker(before).score({"before":before,"after":after,"elapsed_ticks":1,"trace":[after]})
        self.assertAlmostEqual(reward["components"]["damage"],-.03)

    def test_death_zero_potential_but_truncation_retains_bootstrap(self):
        alive,dead = state(),state(1)
        dead["dead"] = True; dead["health"] = 0.; dead["damage"] = 100.
        self.assertGreater(potential(alive,RewardConfig()),0.)
        self.assertEqual(potential(dead,RewardConfig()),0.)
        reward = RewardTracker(alive).score({"before":alive,"after":dead,"elapsed_ticks":1,"trace":[dead]})
        self.assertLess(reward["reward"],-5.)
        death,_ = gae([1.],[2.],[10.],[3],[True],[True])
        cutoff,_ = gae([1.],[2.],[10.],[3],[False],[True])
        self.assertAlmostEqual(death[0],-1.)
        self.assertAlmostEqual(cutoff[0],-1.+.997**3*10.)

    def test_gae_does_not_propagate_across_reset(self):
        advantages,_ = gae([1.,100.],[0.,0.],[2.,0.],[1,1],[False,True],[True,True])
        self.assertAlmostEqual(advantages[0],1.+.997*2.)

    def test_death_after_horizon_does_not_count_as_horizon_death(self):
        a,b,c,d = state(539),state(540),state(541),state(542)
        c["dead"] = True; d["dead"] = True
        original = {"before":a,"after":d,"elapsed_ticks":3,"trace":[b,c,d]}
        clipped = transition_at_horizon(original,540)
        self.assertFalse(clipped["after"]["dead"])
        self.assertEqual(clipped["elapsed_ticks"],1)
        self.assertEqual(len(original["trace"]),3)


if __name__ == "__main__":
    unittest.main()
