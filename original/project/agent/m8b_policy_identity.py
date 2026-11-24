"""Pure M8B checkpoint, target mapping, and schedule identity checks."""
from agent.m8b_contract import VERSION, balance_identity
from agent.m8_policy_identity import PARENT_ACTOR_SHA256, validate_parent_identity

TRAINING_DECISIONS = 49152


def entropy_coefficient(decisions: int) -> float:
    if type(decisions) is not int or decisions < 0:
        raise ValueError("Invalid completed rollout decision count")
    return .002 + (.0002 - .002) * min(decisions / TRAINING_DECISIONS, 1.)


def validate_checkpoint_metadata(metadata, verbs, targets, base_sha, parameters, max_tokens, parent):
    expected_contract = "open_world_hierarchical_v2" if parent else VERSION
    if (len(targets) != 64 or metadata.get("contract") != expected_contract
            or metadata.get("verb_token_ids") != verbs
            or metadata.get("target_token_ids") != targets
            or metadata.get("base_config_sha256") != base_sha
            or metadata.get("base_parameters") != parameters
            or metadata.get("max_tokens") != max_tokens or max_tokens != 1536
            or metadata.get("inference_layout") != "fixed_batch4_left_pad_v1"):
        raise ValueError("Checkpoint encoding/base/layout identity mismatch")
    if not parent:
        expected = balance_identity()
        if (metadata.get("critic_features") != 97
                or metadata.get("reference_actor_sha256") != PARENT_ACTOR_SHA256
                or metadata.get("physical_forward_shape") != [4, max_tokens]
                or type(metadata.get("world_version")) is not int
                or any(metadata.get(key) != value for key, value in expected.items())):
            raise ValueError("Balanced checkpoint rule/reference/critic identity mismatch")


def branch_token_indices(row: dict, verb: int, vocabulary_size: int) -> list[int]:
    indices = row["target_token_indices"][verb]
    if (vocabulary_size != 64 or not indices or len(indices) != len(row["branches"][verb])
            or len(indices) > vocabulary_size or any(type(i) is not int for i in indices)
            or indices != list(range(len(indices)))):
        raise ValueError("Balanced target mapping must preserve original letter positions")
    return list(indices)
