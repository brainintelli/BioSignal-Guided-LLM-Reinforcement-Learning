"""Exploration candidate keeps legal choices and binds shaping to visible facts."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from agent import m6_contract as legacy
from tests.test_m6_semantic_contract import observation as old_observation

ROOT = Path(__file__).resolve().parents[1]


def observation(tick=0, thirst=40., distance=16):
    raw = (ROOT / "configs/m8_exploration_v1.json").read_bytes()
    obs = old_observation()
    obs.update(tick=tick, profile="open3", world_version=5,
               environment_version="m8_exploration_v1", balance_sha256=hashlib.sha256(raw).hexdigest())
    obs["player"]["thirst"] = thirst
    obs["nearby"][1]["path_steps"] = distance
    return obs


def state(obs, dead=False):
    p = obs["player"]
    return dict(tick=obs["tick"], dead=dead, health=p["health"], hunger=p["hunger"],
                thirst=p["thirst"], energy=p["energy"], player={"x":p["x"],"y":p["y"]})


def transition(before, after, dead=False):
    return dict(before=state(before), after=state(after, dead),
                elapsed_ticks=after["tick"]-before["tick"], trace=[state(after, dead)])


class ExplorationContractTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module("agent.m8x_contract")

    def test_identity_and_rules_match_the_new_parameters(self):
        c=self.c;obs=observation();row=c.encode(obs)
        self.assertEqual(row["contract"], "open_world_exploration_v1")
        for expected in ("饱食-.20", "水分-.08", "营地外夜间每步伤.10", "补50水分"):
            self.assertIn(expected, row["messages"][0]["content"])
        for change in ({"profile":"open2"}, {"world_version":4}, {"world_version":True},
                       {"environment_version":"m8_balanced_v1"}, {"balance_sha256":"0"*64}):
            with self.subTest(change=change), self.assertRaises(ValueError):c.encode({**obs,**change})

    def test_all_actions_features_and_branch_codes_are_preserved(self):
        obs=observation();before=copy.deepcopy(obs);row=self.c.encode(obs);old=legacy.encode(obs)
        self.assertEqual(obs,before)
        for key in ("facts","branches","verb_mask","target_rows"):
            self.assertEqual(row[key],old[key])
        self.assertEqual(self.c.critic_features(row),legacy.critic_features(old))
        self.assertEqual(len(self.c.critic_features(row)),97)
        self.assertCountEqual([a for branch in row["branches"] for a in branch],obs["allowed_actions"])
        for i,branch in enumerate(row["branches"]):
            self.assertEqual(row["target_token_indices"][i],list(range(len(branch))))
            if branch:self.assertEqual(self.c.target_messages(row,i)[0],row["messages"][0])

    def test_wrong_target_stage_rules_are_rejected(self):
        row=self.c.encode(observation());row["messages"][0]["content"]=legacy.RULES
        with self.assertRaises(ValueError):self.c.target_messages(row,0)

    def test_visible_water_progress_adds_shaping_without_action_bonuses(self):
        before=observation(0,20.,16);after=observation(1,19.9,8)
        reward=self.c.SurvivalReward(before).score(transition(before,after),after)
        expected=self.c.REWARD_CONFIG["gamma_tick"]*self.c.water_potential(self.c.encode(after)["facts"],dead=False,**self.c.WATER_POTENTIAL)-self.c.water_potential(self.c.encode(before)["facts"],dead=False,**self.c.WATER_POTENTIAL)
        self.assertAlmostEqual(reward["components"]["water_potential"],expected)
        self.assertGreater(expected,0)
        self.assertAlmostEqual(reward["reward"],sum(reward["components"].values()))

    def test_complete_episode_shaping_telescopes_across_day_boundaries(self):
        observations=[observation(t,thirst,distance) for t,thirst,distance in
                      [(1799,20.,16),(1800,19.9,8),(1801,19.8,22),(1802,0.,22)]]
        observations[-1]["player"]["health"]=0.
        observations[-1]["allowed_actions"]=[]
        tracker=self.c.SurvivalReward(observations[0]);gamma=self.c.REWARD_CONFIG['gamma_tick'];total=0.
        for i in range(3):
            after=observations[i+1];r=tracker.score(transition(observations[i],after,dead=i==2),after)
            total+=gamma**i*(r['components']['potential']+r['components']['water_potential'])
        initial=self.c.potential(state(observations[0]))+self.c.water_potential(self.c.encode(observations[0])['facts'],dead=False,**self.c.WATER_POTENTIAL)
        self.assertAlmostEqual(total,-initial,places=12)
        with self.assertRaises(ValueError):tracker.score(transition(observations[-2],observations[-1],True),observations[-1])

    def test_real_duration_and_observation_continuity_are_required(self):
        before=observation(0);after=observation(3,39.8,16);t=transition(before,after)
        with self.assertRaises(ValueError):self.c.SurvivalReward(before).score(t,after)
        t['trace']=[{**state(after),'tick':i} for i in (1,2,3)]
        r=self.c.SurvivalReward(before).score(t,after)
        self.assertAlmostEqual(r['discount'],.9995**3)
        self.assertAlmostEqual(r['components']['survival'],.01*(1+.9995+.9995**2))
        bad=copy.deepcopy(after);bad['tick']=4
        with self.assertRaises(ValueError):self.c.SurvivalReward(before).score(t,bad)
        wrong=copy.deepcopy(t);wrong['before']['thirst']=1
        with self.assertRaises(ValueError):self.c.SurvivalReward(before).score(wrong,after)
        wrong=copy.deepcopy(t);wrong['elapsed_ticks']=True
        with self.assertRaises(ValueError):self.c.SurvivalReward(before).score(wrong,after)

    def test_new_rules_are_exact_and_old_balance_is_rejected(self):
        raw=json.loads((ROOT/'configs/m8_exploration_v1.json').read_text(encoding='utf-8'))
        self.c.validate_balance_config(raw)
        self.assertLess(raw['thirst_drain'],raw['hunger_drain'])
        for change in ({'thirst_drain':.2},{'night_damage':.2},{'world_version':True}):
            with self.assertRaises(ValueError):self.c.validate_balance_config({**raw,**change})


if __name__ == '__main__':unittest.main()
