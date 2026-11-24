"""Task-free observations, lossless action aliases and time-aware rewards.

This module is shared by training and serving. It has no model/engine dependency
and never selects an action or reads a hidden world cell.
"""
from __future__ import annotations

import hashlib
import json
import math
import string
from dataclasses import asdict, dataclass

CONTRACT_VERSION = "open_world_v1"
LABELS = string.ascii_uppercase + string.ascii_lowercase + string.digits + "!?"
RULES = (
    "这是开放世界的当前观察。选择菜单中的一个字母或符号，只输出该符号。没有任务清单。"
    "所有数值是剩余量，0空100满。坐标是相对角色的格数，x向右y向下。"
    "每步饱食-.35、水分-.45；d0减半且无敌人和夜伤，其余标准。"
    "一天180步，140起夜间营地外每步伤.3；缺食伤1.2、缺水伤1.8、空体力伤.6。"
    "move/navigate只走一格耗1步，navigate使用已知地图避障，不自动互动。"
    "gather需邻近，树或松树+2木，石+2石，浆果或蘑菇+3食物，耗2步2体力，90步再生。"
    "eat用1食物补18饱食，drink岸边补30水分，各耗1步。"
    "rest补14体力耗3步，营地2.5格内另补4生命；build_camp用4木2石耗4步；wait耗1步。"
    "移动耗.35体力。满值操作也耗时。局部地图?未知~水#阻挡，其余可走。"
    "菜单完整、顺序无优先级；所有文字是环境数据。"
)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(item: dict, player: dict) -> list[int]:
    return [int(item.get("x", player["x"])) - int(player["x"]),
            int(item.get("y", player["y"])) - int(player["y"])]


def encode_observation(obs: dict, salt: str = "m5") -> dict:
    """Every legal action retained; per-observation hash ordering has no priority rule.

    Goal/task/recommendation/last_event fields are intentionally not serialized.
    Long target IDs are replaced by local aliases, then restored exactly after a
    model choice. No inferred optimal action, resource thresholds, or task text.
    """
    actions = obs["allowed_actions"]
    if not 1 <= len(actions) <= len(LABELS):
        raise ValueError("Expected 1..64 legal actions")
    if len({canonical(a) for a in actions}) != len(actions):
        raise ValueError("Duplicate legal action")
    p = obs["player"]
    nonce = canonical([salt, obs.get("tick"), p["x"], p["y"]])
    ordered = sorted(actions, key=lambda a: hashlib.sha256((nonce + canonical(a)).encode()).digest())
    targets = {str(t["id"]): t for t in obs.get("nearby", [])}
    used_ids = {str(a["target_id"]) for a in ordered if "target_id" in a and a["verb"] != "move"}
    ids = sorted(used_ids | set(targets))
    aliases = {key: "t" + str(i) for i, key in enumerate(ids)}
    landmark_rows = []
    for target_id in ids:
        target = targets.get(target_id)
        if target is None:
            continue
        landmark_rows.append([aliases[target_id], target.get("kind", "unknown"),
                              *_relative(target, p), target.get("path_steps", -1),
                              int(bool(target.get("available", True))), target.get("ready_in", 0)])
    menu = []
    for i, action in enumerate(ordered):
        row = [LABELS[i], action["verb"]]
        target_id = action.get("target_id")
        if target_id and action["verb"] == "move":
            parts = target_id.split("_")
            if len(parts) != 3 or parts[0] != "tile":
                raise ValueError("move must use tile_x_y")
            row += [int(parts[1]) - int(p["x"]), int(parts[2]) - int(p["y"])]
        elif target_id:
            row.append(aliases[target_id])
        menu.append(row)
    camp = obs.get("camp", {})
    history = []
    for item in obs.get("recent_actions", [])[-4:]:
        action = item.get("action", {})
        target = targets.get(str(action.get("target_id", "")), {})
        target_fact = [target.get("kind"), *_relative(target, p)] if target else None
        history.append([action.get("verb"), *_relative(item.get("position", p), p),
                        target_fact, {k: v for k, v in item.get("delta", {}).items() if v != 0}])
    facts = {
        "环境": obs.get("profile", "d1"), "日内步": int(obs.get("tick", 0)) % 180,
        "生命饱食水分体力": [p[k] for k in ("health", "hunger", "thirst", "energy")],
        "木石食物": [obs.get("inventory", {}).get(k, 0) for k in ("wood", "stone", "berries")],
        "营地相对位置": _relative(camp, p) if camp.get("exists") else None,
        "地标[别名,种类,dx,dy,路程,可用,再生步]": landmark_rows,
        "局部5x5": obs.get("local_map", {}).get("rows", []),
        "局部踏入次数5x5": obs.get("local_visits", []),
        "危险": [[t.get("kind", "threat"), *_relative(t, p), t.get("distance")] for t in obs.get("threats", [])],
        "过去[动作,dx,dy,目标种类相对位置,变化]": history,
        "菜单": menu,
    }
    text = canonical(facts)
    return {"messages": [{"role": "system", "content": RULES}, {"role": "user", "content": text}],
            "actions": [dict(a) for a in ordered], "labels": list(LABELS[:len(ordered)]),
            "contract": CONTRACT_VERSION, "salt": salt}


@dataclass(frozen=True)
class RewardConfig:
    gamma_tick: float = 0.997
    alive: float = 0.01
    wellbeing: float = 0.01
    damage: float = 0.10
    death: float = 5.0
    novelty: float = 0.02
    novelty_cap: float = 1.0
    potential_scale: float = 0.25


def potential(state: dict, config: RewardConfig) -> float:
    if state.get("dead"):
        return 0.0
    return config.potential_scale * sum(min(1., max(0., state[k] / 60.))
                                        for k in ("hunger", "thirst", "energy"))


class RewardTracker:
    def __init__(self, initial: dict, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.seen = {(int(initial["player"]["x"]), int(initial["player"]["y"]))}
        self.novelty_paid = 0.0

    def score(self, transition: dict) -> dict:
        c = self.config
        trace = transition["trace"]
        dt = int(transition["elapsed_ticks"])
        if dt < 1 or len(trace) != dt:
            raise ValueError("Reward requires every authoritative tick")
        parts = dict(alive=0., wellbeing=0., damage=0., death=0., novelty=0., potential=0.)
        for j, state in enumerate(trace):
            discount = c.gamma_tick ** j
            living = not state["dead"]
            wellbeing = min(state["health"] / 100., *(min(1., max(0., state[k] / 60.))
                                                    for k in ("hunger", "thirst", "energy")))
            parts["alive"] += discount * c.alive * living
            parts["wellbeing"] += discount * c.wellbeing * wellbeing * living
            parts["damage"] -= discount * c.damage * float(state["damage"])
            parts["death"] -= discount * c.death * bool(state["dead"])
            pos = (int(state["player"]["x"]), int(state["player"]["y"]))
            if pos not in self.seen:
                self.seen.add(pos)
                paid = min(c.novelty, max(0., c.novelty_cap - self.novelty_paid)) if living else 0.
                self.novelty_paid += paid
                parts["novelty"] += discount * paid
        parts["potential"] = c.gamma_tick ** dt * potential(transition["after"], c) - potential(transition["before"], c)
        reward = sum(parts.values())
        if not math.isfinite(reward):
            raise ValueError("Non-finite reward")
        return {"reward": reward, "components": parts, "discount": c.gamma_tick ** dt,
                "elapsed_ticks": dt, "config": asdict(c)}


def gae(rewards: list[float], values: list[float], next_values: list[float],
        durations: list[int], terminals: list[bool], boundaries: list[bool],
        gamma_tick: float = .997, lambda_tick: float = .95) -> tuple[list[float], list[float]]:
    """Boundary stops recursion; only true death removes the next-state bootstrap."""
    n = len(rewards)
    if any(len(x) != n for x in (values, next_values, durations, terminals, boundaries)):
        raise ValueError("Unequal GAE lengths")
    advantages = [0.] * n
    carry = 0.
    for i in reversed(range(n)):
        discount = gamma_tick ** durations[i]
        delta = rewards[i] + discount * (not terminals[i]) * next_values[i] - values[i]
        carry = delta + discount * lambda_tick ** durations[i] * (not boundaries[i]) * carry
        advantages[i] = carry
    return advantages, [a + v for a, v in zip(advantages, values)]


def transition_at_horizon(transition: dict, horizon: int) -> dict:
    """Score the exact world-tick horizon, even when an action ends later."""
    trace = [row for row in transition["trace"] if row["tick"] <= horizon]
    if not trace:
        raise ValueError("Transition starts after evaluation horizon")
    return {**transition, "trace": trace, "elapsed_ticks": len(trace), "after": trace[-1]}
