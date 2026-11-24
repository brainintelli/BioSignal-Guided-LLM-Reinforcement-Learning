import copy
import json
from pathlib import Path
import unittest

from agent import m6_contract as v2
from agent import m6_semantic_contract as v3

ROOT = Path(__file__).resolve().parents[1]


def observation(x=48, y=48):
    return dict(tick=0,profile="open1",player=dict(x=x,y=y,health=100.,hunger=82.,thirst=78.,energy=92.),
        inventory=dict(wood=0,stone=0,berries=2),camp=dict(exists=False,x=x,y=y),
        local_map=dict(rows=["....."]*5),local_visits=[[0]*5 for _ in range(5)],threats=[],recent_actions=[],
        nearby=[dict(id=f"r_{x+2}_{y}",kind="tree",x=x+2,y=y,path_steps=1,available=True,ready_in=0),
                dict(id=f"tile_{x-2}_{y}",kind="water",x=x-2,y=y,path_steps=1,available=True,ready_in=0)],
        allowed_actions=[*[dict(verb="move",target_id=f"tile_{x+dx}_{y+dy}") for dx,dy in v3.DIRECTION_DELTAS],
                         dict(verb="navigate",target_id=f"r_{x+2}_{y}"),
                         dict(verb="navigate",target_id=f"tile_{x-2}_{y}"),
                         dict(verb="gather",target_id=f"r_{x+2}_{y}"),dict(verb="eat"),dict(verb="rest"),dict(verb="wait")])


class SemanticContractTests(unittest.TestCase):
    def test_bijection_full_legal_set_and_original_hash_order(self):
        obs=observation();before=copy.deepcopy(obs)
        old,new=v2.encode(obs),v3.encode(obs)
        self.assertEqual(new["branches"],old["branches"])
        flattened=[action for branch in new["branches"] for action in branch]
        self.assertCountEqual(flattened,obs["allowed_actions"])
        self.assertEqual(obs,before)
        for branch,codes,indices in zip(new["branches"],new["target_codes"],new["target_token_indices"]):
            self.assertEqual(len(branch),len(codes));self.assertEqual(len(codes),len(set(codes)))
            self.assertEqual([new["target_vocab"][i] for i in indices],codes)

    def test_directions_stable_across_positions_and_input_order(self):
        for x,y in [(48,48),(14,65),(83,9)]:
            obs=observation(x,y);obs["allowed_actions"].reverse()
            new=v3.encode(obs);i=v3.VERBS.index("move")
            mapping={tuple(row[1:]):row[0] for row in new["target_rows"][i]}
            self.assertEqual(mapping,v3.DIRECTION_FOR_DELTA)
            self.assertEqual(set(new["target_token_indices"][i]),{64,65,66,67})
            self.assertEqual(new["branches"],v2.encode(obs)["branches"])

    def test_all_features_reconstruct_from_first_stage_and_equal_v2(self):
        obs=observation();encoded=v3.encode(obs)
        facts=json.loads(encoded["messages"][1]["content"])
        from_serialized=dict(facts=facts,verb_mask=[c in facts["合法动作代码"] for c in v3.CODES])
        self.assertEqual(v3.critic_features(from_serialized),v2.critic_features(v2.encode(obs)))
        self.assertEqual(len(v3.critic_features(from_serialized)),97)
        self.assertNotIn("各动作的合法目标",facts)
        self.assertEqual(len(facts["可移动方向[代码,dx,dy]"]),4)

    def test_target_rows_self_contained_without_alias_lookup(self):
        obs=observation();encoded=v3.encode(obs)
        targets={x["id"]:x for x in obs["nearby"]}
        for verb in ("navigate","gather"):
            i=v3.VERBS.index(verb)
            for action,row in zip(encoded["branches"][i],encoded["target_rows"][i]):
                target=targets[action["target_id"]]
                self.assertEqual(row[1:],[target["kind"],target["x"]-48,target["y"]-48,
                                         target["path_steps"],1,target["ready_in"]])
                self.assertFalse(any(isinstance(v,str) and v.startswith("t") and v[1:].isdigit() for v in row))
            message=v3.target_messages(encoded,i)[-1]["content"]
            self.assertIn("种类,dx,dy,已知路程,可用,再生步",message)
        move=v3.target_messages(encoded,v3.VERBS.index("move"))[-1]["content"]
        self.assertIn("方向代码",move);self.assertIn('"up"',move)

    def test_health_and_task_do_not_rank_or_filter_actions(self):
        obs=observation();before=v3.encode(obs)
        obs["player"].update(health=3.,hunger=0.,thirst=1.,energy=2.)
        obs.update(task={"goal":"SECRET_DRINK_NOW"},recommended_action="SECRET_DRINK_NOW")
        obs["tick"]=900
        after=v3.encode(obs)
        for key in ("branches","target_codes","target_token_indices","target_rows","verb_mask"):
            self.assertEqual(before[key],after[key])
        self.assertNotIn("SECRET",str(after["messages"]))

    def test_missing_target_and_nonadjacent_moves_fail(self):
        obs=observation();obs["nearby"].pop()
        with self.assertRaises(ValueError):v3.encode(obs)
        obs=observation();obs["allowed_actions"][0]={"verb":"move","target_id":"tile_60_60"}
        with self.assertRaises(ValueError):v3.encode(obs)

    def test_availability_and_cooldown_are_facts_not_priority(self):
        obs=observation();a=v3.encode(obs)
        obs["nearby"][0].update(available=False,ready_in=31)
        b=v3.encode(obs)
        self.assertEqual(a["branches"],b["branches"])
        self.assertEqual(a["target_codes"],b["target_codes"])
        i=v3.VERBS.index("navigate")
        changed=[r for r in b["target_rows"][i] if r[1]=="tree"]
        self.assertEqual(changed[0][-2:],[0,31])

    def test_same_reward_math_and_independent_version(self):
        self.assertEqual(v3.VERSION,"v3_semantic")
        self.assertNotEqual(v3.VERSION,v2.VERSION)
        self.assertIs(v3.SurvivalReward,v2.SurvivalReward)
        self.assertIs(v3.potential,v2.potential)
        self.assertEqual(len(v3.TARGET_VOCAB),68)
        self.assertEqual(len(set(v3.TARGET_VOCAB)),68)
        self.assertIn("pending",v3.encode(observation())["tokenizer_verification"])

    def test_real_training_records_bijection_facts_and_self_contained_targets(self):
        path=ROOT/"output/agent/m6_training/small_v2_stable/transitions.jsonl"
        if not path.is_file():
            self.skipTest("Real training evidence is not present")
        checked=0
        with path.open(encoding="utf-8") as stream:
            for index,line in enumerate(stream):
                if index%43:continue
                obs=json.loads(line)["observation"]
                old,new=v2.encode(obs),v3.encode(obs)
                self.assertEqual(new["branches"],old["branches"])
                facts=json.loads(new["messages"][1]["content"])
                self.assertEqual(v3.critic_features(dict(facts=facts,verb_mask=new["verb_mask"])),v2.critic_features(old))
                for i,branch in enumerate(new["branches"]):
                    self.assertEqual(len(branch),len(new["target_token_indices"][i]))
                    self.assertEqual([v3.TARGET_VOCAB[t] for t in new["target_token_indices"][i]],new["target_codes"][i])
                    if v3.VERBS[i] in ("navigate","gather"):
                        self.assertTrue(all(len(row)==7 for row in new["target_rows"][i]))
                checked+=1
                if checked==64:break
        self.assertGreaterEqual(checked,32)


if __name__=="__main__":unittest.main()
