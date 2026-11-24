"""Open4 factual Chinese input and separate training-only reward computation.

The actor and critic are constructed from a positive field allowlist. Engine
paths, IDs, reward labels and unknown extra fields never enter either input.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from agent.m8s_state import position, map_state, map_rows, visit_state, visit_rows, time_state, time_tick
from agent.m8b_water_potential import water_potential, shaping_reward
from agent.rl_contract import canonical

VERSION = "open_world_semantic_survival_v3"
PROFILE = "open4"
WORLD_VERSION = 6
ENVIRONMENT_VERSION = "m8_survival_v2"
VERBS = ("move", "navigate", "gather", "eat", "drink", "rest", "build_camp", "wait")
CODES = ("移动", "前往", "采集", "吃饭", "喝水", "休息", "建造", "等待")
KINDS = {"tree": "树木", "pine": "松树", "rock": "岩石", "berry": "浆果", "mushroom": "蘑菇", "water": "饮水岸边", "camp": "营地"}
DIRECTIONS = {(0, -1): "向上", (1, 0): "向右", (0, 1): "向下", (-1, 0): "向左"}
STATE_HEADER = "当前观察：\n"
RULES = (
    "选择合法中文动作及其目标。移动用向上、向下、向左、向右，每次一格；前往地标也只避障走一格，不自动互动。"
    "位置相对自身，上为北，格是距离、刻是时间。周围地形覆盖上下左右各2格共25格；未观察不等于可走，水面和实体不可走，饮水岸边可站立。"
    "地标默认就绪、再生0刻；标记仅观察者不在前往目标中，采集只选标记采集的地标。踏入次数未列为0，历史落点相对现在，未列变化为0。次序和目标数无优先级。"
    "生命、水分、饱食为剩余量，0空100满，生命为0结束。每刻推进后饱食减0.2、水分减0.08；饱食为0伤1.2、水分为0伤1.8。"
    "每天180刻，140起夜间；夜间本身不扣生命，但可出现敌人。营地外敌距小于1.6格每刻伤1.2。"
    "采集耗2刻：树木或松树得2木材，岩石得2石头，浆果或蘑菇得3食物，90刻再生。吃饭用1食物补18饱食，岸边喝水补50水分，各1刻。"
    "休息耗3刻，营地2.5格内且行动前饱食水分均至少20时补2生命；其他地点休息无恢复。建造营地用4木材2石头耗4刻。移动、前往、等待各1刻，满值操作仍耗时。"
)

EXPECTED_BALANCE = dict(
    environment_version=ENVIRONMENT_VERSION, world_version=WORLD_VERSION, profile=PROFILE,
    day_ticks=180, hunger_drain=.2, thirst_drain=.08, drink_restore=50, food_restore=18,
    food_gather_count=3, resource_regrowth_ticks=90, rest_ticks=3, camp_rest_health=2,
    heal_min_hunger=20, heal_min_thirst=20, threat_damage=1.2, hunger_damage=1.2,
    thirst_damage=1.8, initial_health=100, initial_hunger=82, initial_thirst=78,
    initial_berries=2, camp_wood=4, camp_stone=2, camp_radius=2.5,
)


def validate_balance_config(config):
    if not isinstance(config, dict) or set(config) != set(EXPECTED_BALANCE):
        raise ValueError("Survival rule parameter set differs")
    for key, expected in EXPECTED_BALANCE.items():
        value = config[key]
        if isinstance(expected, str):
            valid = isinstance(value, str) and value == expected
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value == expected
        if not valid:
            raise ValueError("Survival rule mismatch: " + key)
    if type(config["world_version"]) is not int:
        raise ValueError("World version must be an integer")


def balance_identity(project=None):
    root = Path(project) if project is not None else Path(__file__).resolve().parents[1]
    raw = (root / "configs/m8_survival_v2.json").read_bytes()
    validate_balance_config(json.loads(raw))
    return dict(profile=PROFILE, world_version=WORLD_VERSION, environment_version=ENVIRONMENT_VERSION,
                balance_sha256=hashlib.sha256(raw).hexdigest())


def validate_observation_identity(obs):
    if type(obs.get("world_version")) is not int or any(obs.get(k) != v for k, v in balance_identity().items()):
        raise ValueError("Observation requires open4/world6 and the exact survival-rule hash")


def _number(value, name, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid finite numeric fact: " + name)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError("Fact outside range: " + name)
    return value


def _integer(value, name):
    if type(value) is not int:
        raise ValueError("Expected integer fact: " + name)
    return value


def _relative(item, player):
    return (_integer(item["x"], "x") - player["x"], _integer(item["y"], "y") - player["y"])


def _landmark_name(item, player):
    if item.get("kind") not in KINDS:
        raise ValueError("Unknown landmark kind; never inject arbitrary prose")
    return KINDS[item["kind"]] + "（" + position(*_relative(item, player)) + "）"


def terrain_text(terrain):
    groups = defaultdict(list)
    for location, kind in terrain["例外位置"].items():
        groups[kind].append(location)
    return "周围地形：其余" + terrain["未列位置均为"] + ("。" if groups else "") + "。".join(
        kind + "：" + "；".join(locations) for kind, locations in groups.items())


def render_scene(state):
    # Spatial facts tend to survive stationary actions; vitals change every tick.
    lines = ["营地：" + (state["营地"] or "无"), terrain_text(state["地形"]), "地标："]
    for obj in state["地标"]:
        flags = []
        if not obj["导航目标"]:
            flags.append("仅观察")
        if obj["采集目标"]:
            flags.append("采集")
        if not obj["可用"]:
            flags.append("未就绪")
        if obj["再生还需刻数"]:
            flags.append("再生" + str(obj["再生还需刻数"]) + "刻")
        lines.append(obj["名称"] + ("：" + "，".join(flags) if flags else ""))
    if not state["地标"]:
        lines.append("无")
    return STATE_HEADER + "\n".join(lines) + "\n动态观察：\n"


def render(state):
    lines = ["威胁：" + ("；".join(f"{t['种类']}（{t['位置']}），距离{t['距离']:g}格" for t in state["威胁"]) or "无"),
              "能移动：" + "、".join(state["合法移动方向"]),
              "合法动作：" + "、".join(state["合法动作"]),
              "踏入次数：" + ("；".join(f"{k}{v}次" for k, v in state["走过的位置"].items()) or "无"),
              "背包：" + "，".join(f"{k}{v}" for k, v in state["背包"].items()),
              "身体：" + "，".join(f"{k}{v:g}" for k, v in state["身体"].items())]
    clock = state["时间"]
    day = "距离入夜还剩刻数" in clock
    lines.append(clock["时段"] + "，距" + ("入夜" if day else "天亮") + str(clock["距离入夜还剩刻数" if day else "距离天亮还剩刻数"]) + "刻")
    lines.append("最近动作：")
    for event in state["最近动作"]:
        changes = "，".join(f"{k}{v:+g}" for k, v in event["变化"].items())
        lines.append(event["动作"] + (event["目标"] or "") + "，落点" + event["完成位置"] + ("，" + changes if changes else ""))
    if not state["最近动作"]:
        lines.append("无")
    return render_scene(state) + "\n".join(lines)


def encode(obs, salt="m6"):
    validate_observation_identity(obs)
    if salt != "m6":
        raise ValueError("Only the original content-hash branch ordering is supported")
    p = obs["player"]
    _integer(p["x"], "x"); _integer(p["y"], "y")
    tick = _integer(obs["tick"], "tick")
    if tick < 0:
        raise ValueError("Negative time")
    vitals = {label: _number(p[key], key, 0, 100) for key, label in (("health", "生命"), ("thirst", "水分"), ("hunger", "饱食"))}
    inventory = {label: _integer(obs["inventory"][key], key) for key, label in (("wood", "木材"), ("stone", "石头"), ("berries", "食物"))}
    if min(inventory.values()) < 0:
        raise ValueError("Negative inventory")
    allowed = obs["allowed_actions"]
    if not 1 <= len(allowed) <= 64 or len({canonical(a) for a in allowed}) != len(allowed):
        raise ValueError("Expected 1..64 unique legal actions")
    for action in allowed:
        verb = action.get("verb")
        if verb not in VERBS or set(action) != ({"verb", "target_id"} if verb in VERBS[:3] else {"verb"}):
            raise ValueError("Invalid legal action shape")
    branches = [sorted((dict(a) for a in allowed if a["verb"] == verb),
                       key=lambda a: hashlib.sha256((salt + canonical(a)).encode()).digest()) for verb in VERBS]
    nearby = {item["id"]: item for item in obs["nearby"]}
    if len(nearby) != len(obs["nearby"]):
        raise ValueError("Duplicate landmark identity")
    names = {key: _landmark_name(item, p) for key, item in nearby.items()}
    if len(set(names.values())) != len(names):
        raise ValueError("Semantic landmark collision")
    targets = []
    for verb, branch in zip(VERBS, branches):
        labels = []
        for action in branch:
            if verb == "move":
                parts = str(action["target_id"]).split("_")
                if len(parts) != 3 or parts[0] != "tile":
                    raise ValueError("Movement requires a tile coordinate")
                delta = (int(parts[1]) - p["x"], int(parts[2]) - p["y"])
                if delta not in DIRECTIONS:
                    raise ValueError("Movement must be one cardinal tile")
                labels.append(DIRECTIONS[delta])
            elif verb in ("navigate", "gather"):
                if action["target_id"] not in names:
                    raise ValueError("Legal target missing an observed semantic descriptor")
                labels.append(names[action["target_id"]])
            else:
                labels.append("")
        if len(set(labels)) != len(labels):
            raise ValueError("Semantic target collision within action")
        targets.append(labels)
    landmarks = []
    navigation = {a["target_id"] for a in branches[1]}
    gathering = {a["target_id"] for a in branches[2]}
    for key in sorted(nearby):
        item = nearby[key]
        ready = _integer(item.get("ready_in", 0), "ready_in")
        if ready < 0 or type(item.get("available", True)) is not bool:
            raise ValueError("Invalid resource availability")
        landmarks.append(dict(名称=names[key], 种类=KINDS[item["kind"]], 位置=position(*_relative(item, p)),
                              可用=item.get("available", True), 再生还需刻数=ready,
                              导航目标=key in navigation, 采集目标=key in gathering))
    history = []
    for event in obs.get("recent_actions", [])[-4:]:
        action = event.get("action", {})
        if action.get("verb") not in VERBS:
            raise ValueError("Unknown history action")
        changes = {label: _number(event.get("delta", {}).get(key, 0), key) for key, label in
                   (("health", "生命"), ("thirst", "水分"), ("hunger", "饱食"))}
        history.append(dict(动作=CODES[VERBS.index(action["verb"])], 完成位置=position(*_relative(event["position"], p)),
                            目标=names.get(action.get("target_id")), 变化={k: v for k, v in changes.items() if v}))
    threats = []
    for threat in obs.get("threats", []):
        threats.append(dict(种类="威胁", 位置=position(*_relative(threat, p)), 距离=_number(threat["distance"], "threat distance", 0)))
    visits = obs["local_visits"]
    if len(visits) != 5 or any(len(row) != 5 for row in visits):
        raise ValueError("Expected complete local visit grid")
    for row in visits:
        for count in row:
            if _integer(count, "visits") < 0:
                raise ValueError("Negative visit count")
    camp = obs.get("camp", {})
    state = dict(营地=position(*_relative(camp, p)) if camp.get("exists") else None,
                 地形=map_state(obs["local_map"]["rows"]), 地标=landmarks, 威胁=threats,
                 合法移动方向=targets[0], 合法动作=[code for code, branch in zip(CODES, branches) if branch],
                 走过的位置=visit_state(visits), 背包=inventory, 身体=vitals, 时间=time_state(tick), 最近动作=history)
    row = dict(messages=[dict(role="system", content=RULES), dict(role="user", content=render(state))],
               branches=branches, verb_mask=[bool(b) for b in branches], target_texts=targets,
               semantic_state=state, contract=VERSION, salt=salt)
    row["features"] = critic_features(row)
    return row


_RESOURCE_KINDS = ("water", "tree", "pine", "rock", "berry", "mushroom")
FEATURE_NAMES = ("health", "thirst", "hunger", "inventory_wood", "inventory_stone", "inventory_food",
                 "day_phase", "day_sin", "day_cos", "camp_exists", "camp_dx", "camp_dy", "threat_count", "nearest_threat")
FEATURE_NAMES += tuple(f"{kind}_{part}" for kind in _RESOURCE_KINDS for part in ("seen", "manhattan_distance", "dx", "dy"))
FEATURE_NAMES += tuple("legal_" + v for v in VERBS)
FEATURE_NAMES += tuple(f"local_walkable_{i}" for i in range(25)) + tuple(f"local_visits_{i}" for i in range(25))
FEATURE_DIM = len(FEATURE_NAMES)


def _position_xy(label):
    if label == "脚下":
        return 0, 0
    x = y = 0
    for part in label.split("、"):
        value = int(part[1:-1])
        if part[0] in "左右":
            x = value if part[0] == "右" else -value
        else:
            y = value if part[0] == "下" else -value
    return x, y


def critic_features(row):
    state = row["semantic_state"]
    result = [state["身体"][k] / 100 for k in ("生命", "水分", "饱食")]
    result += [math.log1p(state["背包"][k]) / 5 for k in ("木材", "石头", "食物")]
    phase = time_tick(state["时间"]) / 180
    result += [phase, math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
    camp = _position_xy(state["营地"]) if state["营地"] else (0, 0)
    result += [float(state["营地"] is not None), camp[0] / 96, camp[1] / 96]
    result += [len(state["威胁"]) / 3, min([t["距离"] for t in state["威胁"]] or [7.]) / 7]
    for kind in _RESOURCE_KINDS:
        candidates = [_position_xy(item["位置"]) for item in state["地标"] if item["种类"] == KINDS[kind]]
        near = min(candidates, key=lambda xy: (abs(xy[0]) + abs(xy[1]), xy)) if candidates else None
        result += [float(near is not None), min(abs(near[0]) + abs(near[1]), 96) / 96 if near else 1.,
                   near[0] / 96 if near else 0., near[1] / 96 if near else 0.]
    result += [float(v) for v in row["verb_mask"]]
    result += [-1. if c == "?" else 0. if c in "~#" else 1. for line in map_rows(state["地形"]) for c in line]
    result += [min(n, 5) / 5 for line in visit_rows(state["走过的位置"]) for n in line]
    if len(result) != 96 or FEATURE_DIM != 96 or not all(math.isfinite(v) for v in result):
        raise ValueError("Three-reserve critic feature contract failed")
    return result


WATER_POTENTIAL = dict(weight=1., urgency_threshold=60., distance_scale=32.)
REWARD_CONFIG = dict(version="survival_potential_three_reserves_v1", gamma_tick=.9995,
                     alive_tick=.01, health_potential=.5, reserve_potential=.25,
                     reserve_saturation=60., water_potential=dict(WATER_POTENTIAL))


def potential(snapshot):
    if snapshot["dead"]:
        return 0.
    return .5 * snapshot["health"] / 100 + .25 * sum(min(1., max(0., snapshot[k] / 60)) for k in ("hunger", "thirst"))


def _training_water(observation, dead=False):
    # Trainer-only adapter preserves the old potential definition. Its path
    # values are not actor/critic state, and no scalar is fed back to encode.
    facts = {"剩余数值": {"水分": observation["player"]["thirst"]},
             "地标[别名,种类,dx,dy,路程,可用,再生步]": [
                 ["", item["kind"], 0, 0, item.get("path_steps", -1)] for item in observation["nearby"]]}
    return water_potential(facts, dead=dead, **WATER_POTENTIAL)


def _check_snapshot(snapshot, observation):
    if type(snapshot.get("tick")) is not int or snapshot["tick"] != observation["tick"] or type(snapshot.get("dead")) is not bool:
        raise ValueError("Reward snapshot time/terminal mismatch")
    p = observation["player"]
    if snapshot["player"] != {"x": p["x"], "y": p["y"]}:
        raise ValueError("Reward snapshot position mismatch")
    for key in ("health", "hunger", "thirst"):
        value = _number(snapshot[key], key, 0, 100)
        if abs(value - p[key]) > .05000001:
            raise ValueError("Reward snapshot differs from rounded observation")


class SurvivalReward:
    """Training signal only: duration-aware survival and telescoping potentials."""
    def __init__(self, initial):
        validate_observation_identity(initial)
        self._observation = deepcopy(initial)
        self._water = _training_water(initial)
        self._closed = False
        self.seen = {(initial["player"]["x"], initial["player"]["y"])}

    def score(self, transition, next_observation):
        if self._closed:
            raise ValueError("A naturally terminated reward tracker cannot continue")
        validate_observation_identity(next_observation)
        before, after, trace = transition["before"], transition["after"], transition["trace"]
        dt = transition["elapsed_ticks"]
        if (type(dt) is not int or dt < 1 or len(trace) != dt or after["tick"] - before["tick"] != dt
                or [item["tick"] for item in trace] != list(range(before["tick"] + 1, after["tick"] + 1))):
            raise ValueError("Complete ordered tick trace required")
        _check_snapshot(before, self._observation)
        _check_snapshot(after, next_observation)
        if before["dead"] or any(type(item.get("dead")) is not bool or item["dead"] for item in trace[:-1]):
            raise ValueError("Reward trace continued past natural death")
        for item in trace:
            for key in ("health", "hunger", "thirst"):
                _number(item[key], key, 0, 100)
        for key in ("tick", "health", "hunger", "thirst", "player", "dead"):
            if trace[-1][key] != after[key]:
                raise ValueError("Final trace differs from successor snapshot")
        terminal = after["dead"]
        if terminal and after["health"] != 0:
            raise ValueError("Natural death requires zero health")
        after_water = _training_water(next_observation, terminal)
        gamma = REWARD_CONFIG["gamma_tick"]
        alive = sum(gamma ** j * REWARD_CONFIG["alive_tick"] * (not item["dead"]) for j, item in enumerate(trace))
        shaped = gamma ** dt * potential(after) - potential(before)
        water = shaping_reward(self._water, after_water, dt, gamma_tick=gamma)
        for item in trace:
            self.seen.add((item["player"]["x"], item["player"]["y"]))
        self._observation, self._water, self._closed = deepcopy(next_observation), after_water, terminal
        return dict(reward=alive + shaped + water, components=dict(survival=alive, potential=shaped, water_potential=water),
                    discount=gamma ** dt, elapsed_ticks=dt, config=deepcopy(REWARD_CONFIG))
