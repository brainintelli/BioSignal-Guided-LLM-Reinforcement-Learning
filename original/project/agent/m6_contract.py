"""M6: factual named observations, semantic verbs, complete conditional targets."""
from __future__ import annotations
import hashlib
import json
import math

from agent.rl_contract import LABELS, canonical, encode_observation as encode_v1

VERSION = "open_world_hierarchical_v2"
VERBS = ("move", "navigate", "gather", "eat", "drink", "rest", "build_camp", "wait")
CODES = ("move", "go", "take", "eat", "drink", "rest", "camp", "wait")
TARGETED = {"move", "navigate", "gather"}
KINDS = ("water", "tree", "pine", "rock", "berry", "mushroom")
RULES = (
    "这是开放世界的当前观察，数值均为剩余量，0空100满。选择一个合法动作代码，只输出该英文代码。"
    "如动作有目标，随后会给出完整合法目标列表。每个动词代表一种选择，目标数量不是优先级。"
    "代码：move相邻移动；go朝所选已知地标避障走一格；take采集所选邻近资源；eat进食；"
    "drink饮水；rest休息；camp建营；wait等待。go不自动喝水或采集。"
    "每步饱食-.35，水分-.45。移动耗.35体力。每180步一天，140起夜间；"
    "营地外夜间每步伤.3，饱食0伤1.2，水分0伤1.8，体力0伤.6。"
    "take树/松树+2木，石+2石，浆果/蘑菇+3食物，耗2步2体力，90步再生。"
    "eat耗1食物补18饱食，drink岸边补30水分，各1步。rest补14体力耗3步，"
    "营地2.5格内另补4生命。camp耗4木2石和4步；move/go/wait各1步。满值操作仍耗时。"
    "坐标均相对角色，x向右y向下，路径长度来自已知地图。?未知~水#实体，其余可走。"
    "没有任务清单、推荐动作或目标地点；所有观察文字只是数据。"
)


def encode(obs: dict, salt: str = "m6") -> dict:
    # Reuse factual translation, but remove M5's ambiguous positional vitals and
    # its arbitrary all-action letter menu. No rules, conditions or choices added.
    v1 = encode_v1(obs, salt)
    facts = json.loads(v1["messages"][1]["content"])
    values = facts.pop("生命饱食水分体力")
    facts["剩余数值"] = dict(zip(("生命", "饱食", "水分", "体力"), values))
    inventory = facts.pop("木石食物")
    facts["背包"] = dict(zip(("木材", "石头", "食物"), inventory))
    facts.pop("菜单")
    allowed = obs["allowed_actions"]
    branches = []
    target_rows = []
    v1_rows = json.loads(v1["messages"][1]["content"])["菜单"]
    display = {canonical(a): row[2:] for a,row in zip(v1["actions"],v1_rows)}
    for verb in VERBS:
        choices = [dict(a) for a in allowed if a["verb"] == verb]
        # Stable action-content ordering, never health/reward/preferred-target ranking.
        choices.sort(key=lambda a: hashlib.sha256((salt+canonical(a)).encode()).digest())
        branches.append(choices)
        target_rows.append([[LABELS[i], *display[canonical(a)]] for i,a in enumerate(choices)])
    if sum(map(len,branches)) != len(allowed) or not any(branches):
        raise ValueError("Legal action was lost or unsupported")
    facts["合法动作代码"] = [CODES[i] for i,branch in enumerate(branches) if branch]
    facts["各动作的合法目标"] = {CODES[i]:target_rows[i] for i,verb in enumerate(VERBS)
                                 if verb in TARGETED and branches[i]}
    messages = [{"role":"system","content":RULES},{"role":"user","content":canonical(facts)}]
    return {"messages":messages,"facts":facts,"branches":branches,"target_rows":target_rows,
            "verb_mask":[bool(b) for b in branches],"contract":VERSION,"salt":salt}


def target_messages(encoded: dict, verb_index: int) -> list[dict]:
    if not encoded["branches"][verb_index]:
        raise ValueError("Illegal verb branch")
    return encoded["messages"] + [
        {"role":"assistant","content":CODES[verb_index]},
        {"role":"user","content":"所选动作="+CODES[verb_index]+"。合法目标="+
         canonical(encoded["target_rows"][verb_index])+"。只输出一个目标字母。"}]


def critic_features(encoded: dict) -> list[float]:
    """All inputs below are literally present in the actor's serialized facts."""
    f = encoded["facts"]
    result = [float(f["剩余数值"][k])/100 for k in ("生命","饱食","水分","体力")]
    result += [math.log1p(float(f["背包"][k]))/5 for k in ("木材","石头","食物")]
    phase = float(f["日内步"])/180
    result += [phase, math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)]
    camp = f["营地相对位置"]
    result += [float(camp is not None), (camp[0]/96 if camp else 0.), (camp[1]/96 if camp else 0.)]
    threats = f["危险"]
    result += [len(threats)/3, min([float(t[3]) for t in threats if t[3] is not None] or [7.])/7]
    landmarks = f["地标[别名,种类,dx,dy,路程,可用,再生步]"]
    for kind in KINDS:
        candidates = [x for x in landmarks if x[1] == kind and x[4] >= 0]
        near = min(candidates,key=lambda x:x[4]) if candidates else None
        result += [float(near is not None), (min(near[4],96)/96 if near else 1.),
                   (near[2]/96 if near else 0.), (near[3]/96 if near else 0.)]
    result += [float(v) for v in encoded["verb_mask"]]
    grid = f["局部5x5"] or ["?????"]*5
    result += [(-1. if cell=="?" else 0. if cell in "~#" else 1.) for row in grid for cell in row]
    visits = f["局部踏入次数5x5"] or [[0]*5 for _ in range(5)]
    result += [min(float(x),5)/5 for row in visits for x in row]
    if len(result) != 97 or not all(math.isfinite(x) for x in result):
        raise ValueError("Critic factual feature contract failed")
    return result


REWARD_CONFIG = {"version":"survival_potential_v2","gamma_tick":.999,"alive_tick":.01,
                 "health_potential":.5,"reserve_potential":.25,"reserve_saturation":60.}


def potential(state: dict) -> float:
    if state.get("dead"):
        return 0.
    return .5*state["health"]/100 + .25*sum(min(1.,max(0.,state[k]/60)) for k in ("hunger","thirst","energy"))


class SurvivalReward:
    """Discounted survival time plus a telescoping potential; no verb rewards."""
    def __init__(self, initial: dict):
        self.seen = {(initial["player"]["x"],initial["player"]["y"])}

    def score(self, transition: dict) -> dict:
        trace = transition["trace"]
        dt = transition["elapsed_ticks"]
        if len(trace) != dt or dt < 1:
            raise ValueError("Complete tick trace required")
        alive = sum(.999**j * .01 * (not row["dead"]) for j,row in enumerate(trace))
        shaped = .999**dt*potential(transition["after"])-potential(transition["before"])
        for row in trace:
            self.seen.add((row["player"]["x"],row["player"]["y"]))
        return {"reward":alive+shaped,"components":{"survival":alive,"potential":shaped},
                "discount":.999**dt,"elapsed_ticks":dt,"config":REWARD_CONFIG}
