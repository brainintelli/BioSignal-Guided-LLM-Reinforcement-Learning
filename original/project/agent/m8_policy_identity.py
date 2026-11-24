"""Pure identity and semantic mapping checks; no model or environment actions."""

PARENT_ACTOR_SHA256 = '02111ea2ce34f61647713d348099de4fc577763bc2423e2a1078097e36dff211'


def entropy_coefficient(decisions: int) -> float:
    if type(decisions) is not int or decisions < 0:
        raise ValueError('Invalid completed decision count')
    return .002 + (.0002-.002)*min(decisions/32768, 1.)


def validate_checkpoint_metadata(metadata, verbs, targets, base_sha, parameters, max_tokens, parent):
    expected_contract = 'open_world_hierarchical_v2' if parent else 'open_world_semantic_v4'
    if (metadata.get('contract') != expected_contract or metadata.get('verb_token_ids') != verbs
            or metadata.get('target_token_ids') != (targets[:64] if parent else targets)
            or metadata.get('base_config_sha256') != base_sha
            or metadata.get('base_parameters') != parameters
            or metadata.get('max_tokens') != max_tokens
            or metadata.get('inference_layout') != 'fixed_batch4_left_pad_v1'):
        raise ValueError('Checkpoint encoding/base/layout identity mismatch')
    if not parent and (metadata.get('critic_features') != 97
                       or metadata.get('reference_actor_sha256') != PARENT_ACTOR_SHA256
                       or metadata.get('physical_forward_shape') != [4,max_tokens]):
        raise ValueError('M8 reference/critic/physical layout mismatch')


def validate_parent_identity(actor_digest: str, reference_digest: str) -> None:
    """At initialization both models must restore the released parent adapter."""
    if actor_digest != PARENT_ACTOR_SHA256 or reference_digest != PARENT_ACTOR_SHA256:
        raise ValueError('Actor initialization/reference must match the released M6 parent')


def branch_token_indices(row: dict, verb: int, vocabulary_size: int) -> list[int]:
    """Map branch action positions to LM token rows without reordering actions."""
    indices = row['target_token_indices'][verb]
    if (not indices or len(indices) != len(row['branches'][verb])
            or any(type(i) is not int or i < 0 or i >= vocabulary_size for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Invalid semantic target mapping')
    return list(indices)
