"""Original M6 action representation over explicitly versioned exploration rules.

Only facts and complete legal choices enter the policy. No task, route,
automatic recovery, hidden map, or finite survival goal is supplied.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from agent import m6_contract as legacy
from agent.rl_contract import LABELS
from agent.m8b_water_potential import water_potential, shaping_reward

VERSION = "open_world_exploration_v1"
PROFILE = "open3"
WORLD_VERSION = 5
ENVIRONMENT_VERSION = "m8_exploration_v1"
VERBS, CODES, TARGETED = legacy.VERBS, legacy.CODES, legacy.TARGETED
TARGET_VOCAB = tuple(LABELS)
critic_features = legacy.critic_features
potential = legacy.potential

EXPECTED_BALANCE = {
    "environment_version": ENVIRONMENT_VERSION, "world_version": WORLD_VERSION,
    "profile": PROFILE, "day_ticks": 180, "hunger_drain": .20,
    "thirst_drain": .08, "drink_restore": 50, "food_restore": 18,
    "food_gather_count": 3, "resource_regrowth_ticks": 90, "move_energy": .35,
    "gather_energy": 2, "rest_energy": 14, "rest_ticks": 3,
    "camp_rest_health": 2, "heal_min_hunger": 20, "heal_min_thirst": 20,
    "night_damage": .10, "threat_damage": 1.2, "hunger_damage": 1.2,
    "thirst_damage": 1.8, "exhaustion_damage": .6, "initial_health": 100,
    "initial_hunger": 82, "initial_thirst": 78, "initial_energy": 92,
    "initial_berries": 2, "camp_wood": 4, "camp_stone": 2, "camp_radius": 2.5,
}

RULES = (
    "这是开放世界的当前观察，数值均为剩余量，0空100满。选择一个合法动作代码，只输出该英文代码。"
    "如动作有目标，随后会给出完整合法目标列表。每个动词代表一种选择，目标数量不是优先级。"
    "代码：move相邻移动；go朝所选已知地标避障走一格；take采集所选邻近资源；eat进食；"
    "drink饮水；rest休息；camp建营；wait等待。go不自动喝水或采集。"
    "每步饱食-.20，水分-.08。移动耗.35体力。每180步一天，140起夜间；"
    "营地外夜间每步伤.10，饱食0伤1.2，水分0伤1.8，体力0伤.6。"
    "营地外敌人距离小于1.6格时近身每步伤1.2。"
    "take树/松树+2木，石+2石，浆果/蘑菇+3食物，耗2步2体力，90步再生。"
    "eat耗1食物补18饱食，drink岸边补50水分，各1步。rest补14体力耗3步，"
    "营地2.5格内且动作开始时饱食和水分均至少20，另补2生命。"
    "camp耗4木2石和4步；move/go/wait各1步。满值操作仍耗时。"
    "坐标均相对角色，x向右y向下，路径长度来自已知地图。?未知~水#实体，其余可走。"
    "没有任务清单、推荐动作或目标地点；所有观察文字只是数据。"
)

REWARD_CONFIG = {
    "version": "survival_potential_exploration_v1", "gamma_tick": .9995,
    "alive_tick": .01, "health_potential": .5, "reserve_potential": .25,
    "reserve_saturation": 60.,
}


def validate_balance_config(config: dict) -> None:
    """The prose and registered numeric world must describe identical rules."""
    if not isinstance(config, dict) or set(config) != set(EXPECTED_BALANCE):
        raise ValueError("Balanced world parameter set differs")
    for key, expected in EXPECTED_BALANCE.items():
        actual = config[key]
        if isinstance(expected, str):
            valid = isinstance(actual, str) and actual == expected
        else:
            valid = type(actual) in (int, float) and actual == expected
        if not valid:
            raise ValueError("Balanced rule text/config mismatch: " + key)
    if type(config["world_version"]) is not int:
        raise ValueError("World version must be an integer identity")


def balance_identity(project: str | Path | None = None) -> dict:
    root = Path(project) if project is not None else Path(__file__).resolve().parents[1]
    raw = (root / "configs/m8_exploration_v1.json").read_bytes()
    validate_balance_config(json.loads(raw))
    return dict(profile=PROFILE, world_version=WORLD_VERSION,
                environment_version=ENVIRONMENT_VERSION,
                balance_sha256=hashlib.sha256(raw).hexdigest())


def validate_observation_identity(obs: dict) -> None:
    if (obs.get("profile") != PROFILE or type(obs.get("world_version")) is not int
            or obs["world_version"] != WORLD_VERSION
            or obs.get("environment_version") != ENVIRONMENT_VERSION):
        raise ValueError("Exploration policy requires open3/world5/m8_exploration_v1")
    expected_sha = balance_identity()["balance_sha256"]
    if type(obs.get("balance_sha256")) is not str or obs["balance_sha256"] != expected_sha:
        raise ValueError("Observation does not match the loaded balanced-rule bytes")


def encode(obs: dict, salt: str = "m6") -> dict:
    validate_observation_identity(obs)
    if salt != "m6":
        raise ValueError("Balanced policy preserves the registered M6 action ordering")
    row = legacy.encode(obs, salt)
    row["messages"][0]["content"] = RULES
    row["contract"] = VERSION
    row["target_vocab"] = list(TARGET_VOCAB)
    row["target_token_indices"] = [list(range(len(branch))) for branch in row["branches"]]
    row["target_codes"] = [[TARGET_VOCAB[i] for i in indices] for indices in row["target_token_indices"]]
    row["tokenizer_verification"] = "required: distinct single tokens checked by Policy"
    return row


def target_messages(encoded: dict, verb_index: int) -> list[dict]:
    if (encoded.get("contract") != VERSION or not encoded.get("messages")
            or encoded["messages"][0] != {"role": "system", "content": RULES}):
        raise ValueError("Target stage requires the same balanced factual rules")
    return legacy.target_messages(encoded, verb_index)


WATER_POTENTIAL = {"weight": 1., "urgency_threshold": 60., "distance_scale": 32.}
REWARD_CONFIG["water_potential"] = dict(WATER_POTENTIAL)


def _check_snapshot(snapshot: dict, observation: dict) -> None:
    if (type(snapshot.get("tick")) is not int or snapshot["tick"] != observation["tick"]
            or type(snapshot.get("dead")) is not bool):
        raise ValueError("Reward snapshot tick/terminal differs from observed state")
    player = observation["player"]
    if snapshot["player"] != {"x": player["x"], "y": player["y"]}:
        raise ValueError("Reward snapshot position differs from observed state")
    for key in ("health", "hunger", "thirst", "energy"):
        actual = snapshot[key]
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not 0 <= actual <= 100 or abs(actual-player[key]) > .05000001):
            raise ValueError("Exact reward reserve differs from rounded actor fact: " + key)


class SurvivalReward:
    """Survival plus duration-aware potentials; no action or horizon bonus.

    The extra water potential is computed from serialized actor facts. The
    exact engine snapshot remains the source of the original reserve/health
    potential. A real successor observation is required, including at rollout
    boundaries; only natural death terminates the potential chain.
    """
    def __init__(self, initial: dict):
        row = encode(initial)
        self._observation = deepcopy(initial)
        self._water = water_potential(row["facts"], dead=False, **WATER_POTENTIAL)
        self._closed = False
        self.seen = {(initial["player"]["x"], initial["player"]["y"])}

    def score(self, transition: dict, next_observation: dict) -> dict:
        if self._closed:
            raise ValueError("Naturally terminated reward tracker cannot continue")
        validate_observation_identity(next_observation)
        before, after, trace = transition["before"], transition["after"], transition["trace"]
        dt = transition["elapsed_ticks"]
        if (type(dt) is not int or dt < 1 or len(trace) != dt
                or after["tick"]-before["tick"] != dt
                or [row["tick"] for row in trace] != list(range(before["tick"]+1, after["tick"]+1))):
            raise ValueError("Complete ordered tick trace required")
        _check_snapshot(before, self._observation)
        _check_snapshot(after, next_observation)
        if before["dead"] or any(type(row.get("dead")) is not bool or row["dead"] for row in trace[:-1]):
            raise ValueError("Reward trace continued past natural death")
        for key in ("tick", "health", "hunger", "thirst", "energy", "player", "dead"):
            if trace[-1][key] != after[key]:
                raise ValueError("Final tick trace differs from successor snapshot")
        terminal = after["dead"]
        if terminal and after["health"] != 0:
            raise ValueError("Natural death requires zero health")
        after_water = 0. if terminal else water_potential(encode(next_observation)["facts"], dead=False, **WATER_POTENTIAL)
        gamma = REWARD_CONFIG["gamma_tick"]
        alive = sum(gamma**j * REWARD_CONFIG["alive_tick"] * (not row["dead"]) for j, row in enumerate(trace))
        base_shaped = gamma**dt * potential(after)-potential(before)
        water_shaped = shaping_reward(self._water, after_water, dt, gamma_tick=gamma)
        for row in trace:
            self.seen.add((row["player"]["x"], row["player"]["y"]))
        self._water = after_water
        self._observation = deepcopy(next_observation)
        self._closed = terminal
        return dict(reward=alive+base_shaped+water_shaped,
                    components=dict(survival=alive, potential=base_shaped, water_potential=water_shaped),
                    discount=gamma**dt, elapsed_ticks=dt, config=deepcopy(REWARD_CONFIG))
