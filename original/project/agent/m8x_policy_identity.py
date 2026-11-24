"""Exploration checkpoint identities; old-world adapters remain reproducible."""
from agent.m8x_contract import VERSION, REWARD_CONFIG, balance_identity
from agent.m8b_policy_identity import branch_token_indices
from agent.m8_policy_identity import PARENT_ACTOR_SHA256, validate_parent_identity


def validate_checkpoint_metadata(metadata, verbs, targets, base_sha, parameters, max_tokens, parent, kernel_identity):
    expected_contract = 'open_world_hierarchical_v2' if parent else VERSION
    if (len(targets) != 64 or metadata.get('contract') != expected_contract
            or metadata.get('verb_token_ids') != verbs or metadata.get('target_token_ids') != targets
            or metadata.get('base_config_sha256') != base_sha
            or metadata.get('base_parameters') != parameters
            or metadata.get('max_tokens') != max_tokens or max_tokens != 1536
            or metadata.get('inference_layout') != 'fixed_batch4_left_pad_v1'):
        raise ValueError('Checkpoint encoding/base/layout identity mismatch')
    if not parent:
        if (metadata.get('critic_features') != 97
                or metadata.get('reference_actor_sha256') != PARENT_ACTOR_SHA256
                or metadata.get('physical_forward_shape') != [4, max_tokens]
                or type(metadata.get('world_version')) is not int
                or any(metadata.get(k) != v for k, v in balance_identity().items())
                or metadata.get('reward_config') != REWARD_CONFIG
                or metadata.get('kernel_identity') != kernel_identity):
            raise ValueError('Exploration checkpoint world/reward/kernel identity mismatch')


TRAINING_DECISIONS = 98304  # Frozen fresh-decision budget.


def entropy_coefficient(decisions):
    if type(decisions) is not int or decisions < 0:
        raise ValueError("Invalid completed rollout count")
    return .002 + (.0002 - .002) * min(decisions / TRAINING_DECISIONS, 1.)
