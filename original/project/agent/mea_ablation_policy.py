"""Matched112D critic and opt-in actor inputs for the three offline arms."""
import json
from pathlib import Path
from agent.mea_ablation import ARMS,encode_arm
from agent.mea_policy import RecordingPolicy
from agent.m8s_trie import choice_tree
from agent.m9_policy import load


def prepare_arm(policy,observations,snapshots):
    if not observations or len(observations)!=len(snapshots):raise ValueError('one input per observation required')
    result=[]
    for obs,snapshot in zip(observations,snapshots):
        if obs['map_mode']!=policy.c.map_mode or obs['navigation_steps']!=policy.c.navigation_steps:
            raise ValueError('environment differs')
        row=encode_arm(obs,snapshot,policy.arm);ids=policy._tokens(row['messages'])
        row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,normalized_targets={},semantic_trees={})
        for v,branch in enumerate(row['branches']):
            if len(branch)>1:
                sequences=[policy.tokenizer.encode(s,add_special_tokens=False) for s in row['target_texts'][v]]
                row['semantic_trees'][v]=choice_tree(sequences,policy.tokenizer.eos_token_id)
        result.append(row)
    return result


class AblationPolicy(RecordingPolicy):
    def __init__(self,c,arm):
        if arm not in ARMS:raise ValueError('unregistered arm')
        self.arm=arm
        super().__init__(c)

    def prepare(self,observations,snapshots):return prepare_arm(self,observations,snapshots)

    def metadata(self):
        return super().metadata()|dict(profile='mea_offline_ablation',ablation_condition=self.arm,
            state2_source='none' if self.arm=='none' else 'recording',
            state2_association='none' if self.arm=='none' else 'within_split_shuffled' if self.arm=='shuffled' else 'sequential_offline')


def load_arm(policy,path):
    meta=json.loads((Path(path)/'policy.json').read_text())
    if meta.get('ablation_condition')!=policy.arm or meta.get('profile')!='mea_offline_ablation':
        raise ValueError('checkpoint condition differs')
    load(policy,path)
