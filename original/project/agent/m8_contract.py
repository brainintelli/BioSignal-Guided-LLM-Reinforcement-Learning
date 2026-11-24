"""M8 semantic representation of the unchanged M6 open1 world.

No task, preferred action, reward change, private state or horizon is added.
The semantic candidate implementation remains shared with its original tests.
Actual tokenizer compatibility is checked by Policy, never asserted here.
"""
from agent import m6_semantic_contract as semantic

VERSION = "open_world_semantic_v4"
VERBS, CODES, TARGETED = semantic.VERBS, semantic.CODES, semantic.TARGETED
TARGET_VOCAB = semantic.TARGET_VOCAB
RULES = semantic.RULES
REWARD_CONFIG = semantic.REWARD_CONFIG
SurvivalReward = semantic.SurvivalReward
potential = semantic.potential
target_messages = semantic.target_messages
critic_features = semantic.critic_features


def encode(obs: dict, salt: str = "m6") -> dict:
    if obs.get("profile") != "open1":
        raise ValueError("M8 is registered only for the unchanged open1 world")
    version = obs.get("world_version", 2)
    if type(version) is not int or version != 2:
        raise ValueError("M8 cannot silently enable a rejected environment challenge")
    encoded = semantic.encode(obs, salt)
    encoded["contract"] = VERSION
    encoded["tokenizer_verification"] = "required: distinct single tokens checked by Policy"
    return encoded
