"""Warm-started actor/critic; only LoRA and critic are trainable."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from agent.m9_policy import Policy as BasePolicy, load
from agent.mea_policy import RecordingPolicy
from agent.m8s_trie import choice_tree
from tools.mea_offline import CONFIG as BASE_CONFIG
from .adapter import CONTRACT, encode
from .features import NAMES
from .protocol import cell, BETA, PROFILE

CONFIG = BASE_CONFIG | dict(minibatch=128, critic_minibatch=128, activation_offload='cpu_pinned_preserve_strides')


class Policy(RecordingPolicy):
    def __init__(self, config, arm, normalization_id):
        self.arm, self.normalization_id = arm, normalization_id
        super().__init__(config)

    def prepare(self, observations, snapshots):
        if not observations or len(observations) != len(snapshots):
            raise ValueError('One finite snapshot per observation required')
        result = []
        for obs, snapshot in zip(observations, snapshots):
            row = encode(obs, snapshot, self.arm, self.normalization_id)
            ids = self._tokens(row['messages'])
            row.update(input_ids=ids, kv_prefix_ids=ids, scene_prefix_ids=ids, normalized_targets={}, semantic_trees={})
            for v, branch in enumerate(row['branches']):
                if len(branch)>1:
                    sequences=[self.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]]
                    row['semantic_trees'][v]=choice_tree(sequences,self.tokenizer.eos_token_id)
            result.append(row)
        return result

    def values(self, prepared):
        if not prepared or any(r.get('contract') != CONTRACT or len(r['critic_input']) != 112 for r in prepared):
            raise ValueError('New observation contract required')
        facts=torch.tensor([r['critic_input'] for r in prepared],device=self.device_name,dtype=torch.float32)
        return torch.cat([self.critic(facts[i:i+1]) for i in range(len(prepared))]).squeeze(-1)

    def metadata(self):
        meta = super().metadata()
        return meta | dict(contract=CONTRACT, profile='mea_state2_r2_digital', arm=self.arm,
            critic_feature_names=meta['critic_feature_names'][:108]+['mea_v2_'+k for k in NAMES],
            normalization_id=self.normalization_id, state2_source='masked' if self.arm in ('A','B') else 'historical_recording',
            state2_association='historical_no_game_causality', r2_enabled=self.arm in ('B','C','D'),
            r2_source='simulation_from_game_state' if self.arm in ('B','C','D') else 'disabled', beta=BETA,
            stimulation=False, realtime_training=False, stimulus_profile_sha256=PROFILE.digest)


def new_policy(model_config, arm, block, normalization_id, batch_size=128):
    config=cell(block,arm)
    torch.manual_seed(config['initial_seed'])
    paths=json.loads(Path(model_config).read_text())
    if set(paths) != {'model','initial_checkpoint','parent_checkpoint','warm_checkpoint'}:
        raise ValueError('Unexpected model paths')
    c=SimpleNamespace(**(CONFIG | dict(minibatch=batch_size,critic_minibatch=batch_size)),
                      **paths,rng_seed=config['initial_seed'])
    if arm=='A0':
        return BasePolicy(c)
    return Policy(c,arm,normalization_id)


def load_checked(policy, path):
    meta=json.loads((Path(path)/'policy.json').read_text())
    for key in ('contract','arm','normalization_id','critic_feature_names','r2_enabled','r2_source','beta','stimulus_profile_sha256'):
        if meta.get(key) != policy.metadata().get(key):
            raise ValueError('Checkpoint version/binding mismatch: '+key)
    load(policy,path)
