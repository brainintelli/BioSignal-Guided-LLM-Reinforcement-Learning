"""Isolated M9 policy extension; original shared modules remain untouched."""
import json
from pathlib import Path
import torch
from agent.m9_policy import Policy as M9Policy,load as load_m9,verify_checkpoint
from agent.m8s_trie import choice_tree
from sr_contract import encode,CONTRACT
from sr_protocol import ARMS,STATE_ARMS,REWARD_ARMS,STATE_FEATURES
EXPECTED='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'

def expand_critic(critic):
    old=critic[0]
    if tuple(old.weight.shape)!=(128,108):raise ValueError('original108D critic required')
    new=torch.nn.Linear(112,128,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():new.weight.zero_();new.weight[:,:108].copy_(old.weight);new.bias.copy_(old.bias)
    critic[0]=new

def consistent_values(policy,prepared):
    if not prepared or any(len(p['critic_input'])!=112 for p in prepared):raise ValueError('112D input required')
    facts=torch.tensor([p['critic_input'] for p in prepared],device=policy.device_name,dtype=torch.float32)
    return torch.cat([policy.critic(facts[i:i+1]) for i in range(len(prepared))]).squeeze(-1)

class Policy(M9Policy):
    def __init__(self,c,arm,dataset_pin):
        if arm not in ARMS or verify_checkpoint(c.warm_checkpoint)['actor_sha256']!=EXPECTED:raise ValueError('arm/originalcheckpoint identity')
        self.arm=arm;self.dataset_pin=dataset_pin
        super().__init__(c,reference=True)
        if self.digest()!=EXPECTED or self.reference_digest()!=EXPECTED:raise ValueError('originalactor/reference changed')
        expand_critic(self.critic);self.critic_initialization='original108_to112_zero_new_columns'
    def prepare(self,observations,snapshots):
        if not observations or len(observations)!=len(snapshots):raise ValueError('paired snapshots required')
        result=[]
        for obs,s in zip(observations,snapshots):
            if s['dataset_sha256']!=self.dataset_pin or obs['map_mode']!=self.c.map_mode or obs['navigation_steps']!=self.c.navigation_steps:raise ValueError('data/world identity')
            row=encode(obs,s,self.arm);ids=self._tokens(row['messages']);row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,normalized_targets={},semantic_trees={})
            for v,branch in enumerate(row['branches']):
                if len(branch)>1:
                    seq=[self.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]];row['semantic_trees'][v]=choice_tree(seq,self.tokenizer.eos_token_id)
            result.append(row)
        return result
    def values(self,prepared):return consistent_values(self,prepared)
    def metadata(self):
        result=super().metadata();result.update(base_contract=result['contract'],contract=CONTRACT,profile='28073_offline_state_r1only_exploratory',arm=self.arm,
            dataset_sha256=self.dataset_pin,critic_features=112,critic_feature_names=result['critic_feature_names']+['mea_z_'+x for x in STATE_FEATURES],
            state2_enabled=self.arm in STATE_ARMS,r2_enabled=False,r2_biologically_validated=False,
            reward_contract='r1_only_v1',r1_weight=1.,aux_weight=0.,aux_enabled=False,
            stimulation=False,source_stimulation='recorded_in_manifest_0_to_2s_excluded',realtime_training=False,training_scope_qc_passed=True,source_full_qc_passed=False,source_association='prerecorded_action_independent',critic_execution='per_row_fp32')
        return result

def load(policy,path):
    meta=json.loads((Path(path)/'policy.json').read_text(encoding='utf-8'))
    for k,v in policy.metadata().items():
        if k in ('arm','dataset_sha256','reward_contract','aux_weight','aux_enabled','r2_enabled','state2_enabled','r1_weight','source_association','source_full_qc_passed') and meta.get(k)!=v:raise ValueError('checkpointcondition '+k)
    load_m9(policy,path)
