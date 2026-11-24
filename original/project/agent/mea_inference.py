"""Opt-in inference bridge for an already loaded, exclusively owned M9 actor.

No model loading, optimizer, checkpoint writes, or device access occurs here.
The default simulator uses State2Probe instead; it does not claim a Qwen run.
"""
from copy import deepcopy

from agent.m8s_trie import choice_tree
from agent.mea_closed_loop import ACTOR_FIELDS
from agent.mea_snapshot import vector


def prepare_inference_row(policy, projection):
    if not isinstance(projection, dict) or set(projection) != set(ACTOR_FIELDS):
        raise ValueError('Actor input must be the explicit state1/state2 projection')
    vector(projection['wetware_features'], 'state2')
    row = deepcopy(projection)
    ids = policy._tokens(row['messages'])
    row.update(input_ids=ids, kv_prefix_ids=ids, scene_prefix_ids=ids, normalized_targets={}, semantic_trees={})
    for verb, branch in enumerate(row['branches']):
        if len(branch) > 1:
            sequences = [policy.tokenizer.encode(text, add_special_tokens=False) for text in row['target_texts'][verb]]
            row['semantic_trees'][verb] = choice_tree(sequences, policy.tokenizer.eos_token_id)
    return row


class FrozenM9Actor:
    action_source = 'frozen_m9_inference'

    def __init__(self, policy):
        import torch
        for key, value in dict(map_mode='infinite_freeze', navigation_steps=4, state_variant='reserves').items():
            if getattr(policy.c, key) != value: raise ValueError('M9 actor identity mismatch: ' + key)
        if any(p.grad is not None for p in policy.model.parameters()):
            raise ValueError('Use a separate clean inference policy, not a training instance')
        self.policy, self.torch = policy, torch
        policy.model.eval()
        self.actor_digest = policy.digest()

    def choose(self, projection):
        policy = self.policy
        if policy.digest() != self.actor_digest or policy.model.training:
            raise ValueError('Actor changed since inference session start')
        row = prepare_inference_row(policy, projection)
        with self.torch.inference_mode():
            verb = int(policy.verb_distribution([row]).probs.argmax(-1).item())
            target = int(policy.target_distribution([row], [verb]).probs.argmax(-1).item())
        if policy.digest() != self.actor_digest or any(p.grad is not None for p in policy.model.parameters()):
            raise ValueError('Inference changed actor weights or gradients')
        return deepcopy(row['branches'][verb][target])
