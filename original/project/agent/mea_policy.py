"""Opt-in real M9 actor/critic adapter for audited offline recordings only."""
import json
from pathlib import Path
import torch
from agent.m9_policy import Policy as M9Policy,verify_checkpoint
from agent.m8s_trie import choice_tree
from agent.mea_contract import encode
from agent.mea_replay import RecordingSnapshot,RECORDING_CONTRACT
from agent.mea_snapshot import FEATURE_NAMES

EXPECTED_ACTOR='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'


def expand_recording_critic(critic):
    old=critic[0]
    if tuple(old.weight.shape)!=(128,108):raise ValueError('expected 108-D M9 critic')
    new=torch.nn.Linear(112,128,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:108].copy_(old.weight);new.bias.copy_(old.bias)
    critic[0]=new


def prepare_recordings(policy,observations,snapshots):
    if not observations or len(observations)!=len(snapshots):raise ValueError('one recording snapshot per observation required')
    result=[]
    for obs,snapshot in zip(observations,snapshots):
        if not isinstance(snapshot,RecordingSnapshot):raise ValueError('offline recording snapshot required')
        if obs['map_mode']!=policy.c.map_mode or obs['navigation_steps']!=policy.c.navigation_steps:
            raise ValueError('environment changed')
        row=encode(obs,snapshot,policy.c.state_variant);ids=policy._tokens(row['messages'])
        row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,normalized_targets={},semantic_trees={})
        for v,branch in enumerate(row['branches']):
            if len(branch)>1:
                sequences=[policy.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]]
                row['semantic_trees'][v]=choice_tree(sequences,policy.tokenizer.eos_token_id)
        result.append(row)
    return result


class RecordingPolicy(M9Policy):
    def __init__(self,c,reference=True):
        manifest=verify_checkpoint(c.warm_checkpoint)
        if manifest['actor_sha256']!=EXPECTED_ACTOR:raise ValueError('wrong shared combined_0064 actor')
        warm=json.loads((Path(c.warm_checkpoint)/'policy.json').read_text())
        for key,expected in dict(contract='m9_research_v1',map_mode='infinite_freeze',navigation_steps=4,
            state_variant='reserves',reward_variant='bottleneck',gamma_tick=.99985,critic_features=108).items():
            if warm.get(key)!=expected:raise ValueError('warm checkpoint identity: '+key)
        for key in ('map_mode','navigation_steps','state_variant','reward_variant','gamma_tick'):
            if getattr(c,key)!=warm[key]:raise ValueError('offline config differs from warm policy: '+key)
        super().__init__(c,reference=reference)
        expand_recording_critic(self.critic)
        self.critic_initialization='combined_0064_108_to112_new_columns_zero'

    def prepare(self,observations,snapshots):return prepare_recordings(self,observations,snapshots)

    def values(self,prepared):
        if not prepared or any(r.get('contract')!=RECORDING_CONTRACT or len(r['critic_input'])!=112 for r in prepared):
            raise ValueError('offline critic input contract')
        facts=torch.tensor([r['critic_input'] for r in prepared],device=self.device_name,dtype=torch.float32)
        # Serial collection and batched PPO must use the same FP32 reduction
        # shape. These tiny MLP calls retain autograd, unlike cached values.
        return torch.cat([self.critic(facts[i:i+1]) for i in range(len(prepared))]).squeeze(-1)

    def metadata(self):
        result=super().metadata()
        result.update(base_contract=result['contract'],contract=RECORDING_CONTRACT,profile='mea_offline',
            critic_features=112,critic_feature_names=result['critic_feature_names']+['mea_'+k for k in FEATURE_NAMES],
            state2_source='recording',state2_association='sequential_offline',stimulation=False,
            critic_execution_layout='per_row_fp32',
            alpha=1.,r2_enabled=False,realtime_training=False)
        return result
