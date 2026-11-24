"""Pending semantic-target interface. NOT connected to the running M6 policy.

Preserves every v2 legal action and its stable hash order, with no action choice,
reward modification, macro action or state-conditioned candidate preference.
Direction words still require single-token verification on the actual tokenizer.
"""
from __future__ import annotations

import copy

from agent import m6_contract as v2
from agent.rl_contract import LABELS, canonical

VERSION = "v3_semantic"
VERBS, CODES, TARGETED = v2.VERBS, v2.CODES, v2.TARGETED
DIRECTION_CODES = ("up", "right", "down", "left")
DIRECTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIRECTION_FOR_DELTA = dict(zip(DIRECTION_DELTAS, DIRECTION_CODES))
TARGET_VOCAB = tuple(LABELS) + DIRECTION_CODES
TARGET_VOCAB_INDEX = {code: index for index, code in enumerate(TARGET_VOCAB)}
RULES = v2.RULES + (
    "第一阶段只输出动作代码。若选择move，第二阶段用up/right/down/left表示上/右/下/左，"
    "方向不会随位置改变。若选择go或take，第二阶段每个目标代码同行列出种类、相对位置、"
    "已知路程、可用与再生步，不需要另查目标别名。路程-1表示未提供或不可达信息。"
    "候选顺序不是建议；没有任何动作由接口自动选择。"
)

# Math is shared explicitly; the candidate changes representation only.
REWARD_CONFIG = copy.deepcopy(v2.REWARD_CONFIG)
SurvivalReward = v2.SurvivalReward
potential = v2.potential


def encode(obs: dict, salt: str = "m6") -> dict:
    previous = v2.encode(obs, salt)
    facts = copy.deepcopy(previous["facts"])
    facts.pop("各动作的合法目标")
    branches = copy.deepcopy(previous["branches"])
    targets = {str(target["id"]): target for target in obs.get("nearby", [])}
    player = obs["player"]
    target_rows, target_codes, target_token_indices = [], [], []
    for verb, actions in zip(VERBS, branches):
        rows, codes = [], []
        for index, action in enumerate(actions):
            if verb == "move":
                parts = str(action.get("target_id", "")).split("_")
                if len(parts) != 3 or parts[0] != "tile":
                    raise ValueError("Semantic move requires tile_x_y")
                dx, dy = int(parts[1])-int(player["x"]), int(parts[2])-int(player["y"])
                if (dx, dy) not in DIRECTION_FOR_DELTA:
                    raise ValueError("Semantic move must be one orthogonal adjacent tile")
                code = DIRECTION_FOR_DELTA[(dx, dy)]
                rows.append([code, dx, dy])
            elif verb in ("navigate", "gather"):
                target = targets.get(str(action.get("target_id", "")))
                if target is None:
                    raise ValueError("Legal target has no factual landmark description")
                code = LABELS[index]
                rows.append([code, target.get("kind", "unknown"),
                    int(target["x"])-int(player["x"]), int(target["y"])-int(player["y"]),
                    target.get("path_steps", -1), int(bool(target.get("available", True))),
                    target.get("ready_in", 0)])
            else:
                # A singleton branch is deterministic, with no second LM choice.
                code = LABELS[index]
                rows.append([code])
            codes.append(code)
        if len(set(codes)) != len(codes):
            raise ValueError("Target codes must be unique within a verb branch")
        target_rows.append(rows)
        target_codes.append(codes)
        target_token_indices.append([TARGET_VOCAB_INDEX[code] for code in codes])
    facts["可移动方向[代码,dx,dy]"] = copy.deepcopy(target_rows[VERBS.index("move")])
    return dict(messages=[{"role": "system", "content": RULES},
                          {"role": "user", "content": canonical(facts)}],
                facts=facts, branches=branches, target_rows=target_rows,
                target_codes=target_codes, target_token_indices=target_token_indices,
                target_vocab=list(TARGET_VOCAB), verb_mask=list(previous["verb_mask"]),
                contract=VERSION, salt=salt,
                tokenizer_verification="pending: every TARGET_VOCAB entry must be a distinct single token")


def target_messages(encoded: dict, verb_index: int) -> list[dict]:
    if not 0 <= verb_index < len(VERBS) or not encoded["branches"][verb_index]:
        raise ValueError("Illegal verb branch")
    verb = VERBS[verb_index]
    if verb == "move":
        fields = "[方向代码,dx,dy]"
        instruction = "只输出本列表中的一个方向代码。"
    elif verb in ("navigate", "gather"):
        fields = "[目标代码,种类,dx,dy,已知路程,可用,再生步]"
        instruction = "只输出本列表中的一个目标字母。"
    else:
        fields = "[唯一选项]"
        instruction = "此动作没有目标选择。"
    return encoded["messages"] + [
        {"role": "assistant", "content": CODES[verb_index]},
        {"role": "user", "content": "所选动作="+CODES[verb_index]+"。合法目标"+fields+"="+
         canonical(encoded["target_rows"][verb_index])+"。"+instruction},
    ]


def critic_features(encoded: dict) -> list[float]:
    """Same 97 numbers as v2; no private observation, target choice or oracle."""
    return v2.critic_features({"facts": encoded["facts"], "verb_mask": encoded["verb_mask"]})
