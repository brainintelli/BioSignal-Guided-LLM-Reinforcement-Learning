"""Real native transport plus pure contract checks for the M9 research campaign."""
from pathlib import Path
import importlib
import unittest
import math
import copy
ROOT=Path(__file__).resolve().parents[1]
GODOT=ROOT/".tools/godot/Godot_v4.7.2-stable_win64.exe"
import os
GODOT=Path(os.environ.get("MEADOW_GODOT",str(GODOT)))

class M9Contracts(unittest.TestCase):
    def modules(self):
        try:
            return importlib.import_module("agent.m9_contract"),importlib.import_module("agent.m9_env")
        except ModuleNotFoundError:
            self.fail("M9 contract and native transport are not implemented")
    def test_factual_state_does_not_leak_reward_or_recommendation(self):
        c,e=self.modules()
        with e.GodotEnv(str(GODOT),str(ROOT),map_mode="finite",navigation_steps=1) as env:
            obs=env.reset(3100001)["observation"]
            row=c.encode(obs,"reserves")
            self.assertIn("不补充时饱食耗尽",row["messages"][1]["content"])
            poisoned=copy.deepcopy(obs);poisoned["reward"]=100000;poisoned["recommendation"]="喝水"
            self.assertEqual(row,c.encode(poisoned,"reserves"))
            self.assertEqual(row["branches"],c.encode(obs,"legacy")["branches"])
            bad=copy.deepcopy(obs);bad["world_version"]=6
            with self.assertRaises(ValueError):c.encode(bad,"legacy")
    def test_rewards_duration_discount_and_death_at_horizon(self):
        c,e=self.modules()
        with e.GodotEnv(str(GODOT),str(ROOT),map_mode="finite",navigation_steps=4) as env:
            obs=env.reset(3100002)["observation"]
            rules=c.encode(obs)["messages"][0]["content"]
            self.assertNotIn("移动、前往、等待各1刻",rules)
            self.assertIn("前往每格1刻",rules)
            self.assertEqual(obs["navigation_assistance"],"selected_target_up_to_4_collision_safe_steps")
            reward=c.Reward(obs,variant="survival",gamma=.9)
            nxt=env.step({"verb":"rest"})
            value=reward.score(nxt["transition"],nxt["observation"])
            self.assertAlmostEqual(value["reward"],.01*(1+.9+.81),places=12)
            from agent.m9_metrics import summarize
            episodes=[dict(seed=1,replicate=0,survival_ticks=5400,terminal=True,alive_at_horizon=False,alive_at_1800=True)]
            self.assertEqual(summarize(episodes)["successes"],0)
            self.assertEqual(summarize(episodes)["successes_10day"],1)
    def test_native_replay_and_macro_time_accounting(self):
        c,e=self.modules()
        with e.GodotEnv(str(GODOT),str(ROOT),map_mode="infinite_freeze",navigation_steps=4) as a,e.GodotEnv(str(GODOT),str(ROOT),map_mode="infinite_freeze",navigation_steps=4) as b:
            oa=a.reset(3100003)["observation"];ob=b.reset(3100003)["observation"]
            self.assertEqual(oa,ob)
            for i in range(30):
                acts=oa["allowed_actions"]
                action=next((x for x in acts if x["verb"]=="navigate"),{"verb":"wait"})
                ra=a.step(action);rb=b.step(action)
                self.assertEqual(ra,rb)
                self.assertEqual(ra["transition"]["elapsed_ticks"],len(ra["transition"]["trace"]))
                self.assertGreaterEqual(ra["transition"]["elapsed_ticks"],1)
                self.assertLessEqual(ra["transition"]["elapsed_ticks"],4)
                oa=ra["observation"]
    def test_bottleneck_reward_is_telescoping_not_action_bonus(self):
        c,e=self.modules()
        with e.GodotEnv(str(GODOT),str(ROOT),map_mode="finite",navigation_steps=1) as env:
            obs=env.reset(3100004)["observation"];gamma=.9995
            reward=c.Reward(obs,"bottleneck",gamma)
            initial=c.potential(obs,"bottleneck",False)
            total=0.;living=0.;ticks=0
            for _ in range(4):
                nxt=env.step({"verb":"wait"})
                row=reward.score(nxt["transition"],nxt["observation"])
                total+=gamma**ticks*row["reward"]
                living+=gamma**ticks*.01
                ticks+=1;obs=nxt["observation"]
            self.assertAlmostEqual(total-living,gamma**4*c.potential(obs,"bottleneck",False)-initial,places=10)
            self.assertEqual(c.potential(obs,"bottleneck",True),0.)
    def test_selection_never_uses_heldout_and_rejects_incomplete_pair(self):
        self.modules()
        from agent.m9_metrics import select_arm
        rows=[dict(arm="b",replicate=i,summary=dict(successes=1,cvar20=3000,rmst=4000)) for i in (0,1)]
        rows += [dict(arm="a",replicate=i,summary=dict(successes=2,cvar20=3500,rmst=4500)) for i in (0,1)]
        self.assertEqual(select_arm(rows,["b","a"]),["a","b"])
        with self.assertRaises(ValueError):select_arm(rows[:-1],["b","a"])

    def test_more_food_never_reduces_bottleneck_potential(self):
        c,e=self.modules()
        with e.GodotEnv(str(GODOT),str(ROOT),map_mode="finite",navigation_steps=1) as env:
            obs=env.reset(2304)["observation"]
            obs["player"].update(hunger=20.,thirst=20.)
            obs["inventory"]["berries"]=0
            before=c.potential(obs,"bottleneck")
            obs["inventory"]["berries"]=6
            self.assertGreaterEqual(c.potential(obs,"bottleneck"),before)

    def test_time_aware_returns_use_actual_macro_duration(self):
        try:
            from agent.m9_run import finish_trajectory
        except ModuleNotFoundError:
            self.fail("M9 PPO entry is not implemented")
        row=dict(value=2.,reward=.2,duration=4,terminal=False,boundary=False,next_obs={})
        result=finish_trajectory([row],3.,.5,.9)
        self.assertAlmostEqual(result[0]["return"],.3875)
        dead=dict(value=2.,reward=.2,duration=4,terminal=True,boundary=True,next_obs={})
        self.assertAlmostEqual(finish_trajectory([dead],99.,.5,.9)[0]["return"],.2)

    def test_final_gate_needs_95_and_exact_ten_day_regression(self):
        from agent.m9_metrics import compare
        def episodes(successes):
            return [dict(seed=i,replicate=0,survival_ticks=5400 if i<successes else 2000,
                terminal=i>=successes,alive_at_horizon=i<successes,alive_at_1800=True) for i in range(100)]
        self.assertFalse(compare(episodes(94),episodes(94))["numerical_gate_passed"])
        result=compare(episodes(95),episodes(94))
        self.assertTrue(result["numerical_gate_passed"])
        self.assertFalse(result["milestone_achieved"])

if __name__=="__main__":unittest.main()

