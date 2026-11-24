"""Canonical B4 policy for registered M9 interventions; no executor chooses goals."""
import json
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from agent.m8s3_batch_policy import CanonicalBatchPolicy
from agent.m8s3_supply import expand_critic,readiness_features,READINESS_NAMES
from agent.m8s3_run import exploration_distribution
from agent.m8s3_training import sha,write
from agent.m8s_trie import choice_tree
from agent.m9_contract import encode,VERSION
from agent.m9_context import context_layout,tokenize_observation

IDENTITY_KEYS=("contract","world_version","map_mode","navigation_steps","state_variant","reward_variant",
               "gamma_tick","verb_exploration_epsilon","base_config_sha256","kernel_identity","critic_feature_names")

def verify_checkpoint(path):
    path=Path(path)
    manifest=json.loads((path/"manifest.json").read_text())
    for name,digest in manifest["files"].items():
        if sha(path/name)!=digest:raise ValueError("Checkpoint changed: "+name)
    return manifest

class Policy(CanonicalBatchPolicy):
    def __init__(self,c,reference=True):
        self.c=c
        super().__init__(c.model,checkpoint=c.initial_checkpoint,parent_checkpoint=c.parent_checkpoint,
            seed=c.rng_seed,reset_critic=True,with_reference=reference,max_tokens=1536,prefix_layout="system_only")
        # The historical warm checkpoints were loaded under their original
        # identity above. Only the rejection limit and overflow buckets expand.
        self.bucket_boundaries=context_layout(c.max_tokens)
        self.max_tokens=c.max_tokens
        manifest=verify_checkpoint(c.warm_checkpoint)
        self.anchor_digest=manifest["actor_sha256"]
        weights=load_file(str(Path(c.warm_checkpoint)/"adapter/adapter_model.safetensors"))
        set_peft_model_state_dict(self.model,weights)
        if self.digest()!=self.anchor_digest:raise ValueError("Warm actor differs")
        expand_critic(self.critic)
        self.critic.load_state_dict(torch.load(Path(c.warm_checkpoint)/"critic.pt",map_location=self.device_name,weights_only=True),strict=True)
        self.critic_initialization="explicit_unreleased_supply3_48_warm108"
        if reference:
            set_peft_model_state_dict(self.reference_model,weights)
            self.reference_model.requires_grad_(False);self.reference_model.eval()
            self.assert_reference_frozen()
        self.execution_cache.revisions.clear();self.clear_prefix_cache()
    def _tokens(self,messages):
        return tokenize_observation(self.tokenizer,messages,self.max_tokens)
    def prepare(self,observations,salt="m6"):
        if salt!="m6":raise ValueError("Unregistered branch salt")
        result=[]
        for obs in observations:
            if obs["map_mode"]!=self.c.map_mode or obs["navigation_steps"]!=self.c.navigation_steps:
                raise ValueError("Environment differs from registered arm")
            row=encode(obs,self.c.state_variant)
            ids=self._tokens(row["messages"])
            row.update(input_ids=ids,kv_prefix_ids=ids,scene_prefix_ids=ids,
                       normalized_targets={},semantic_trees={})
            for v,branch in enumerate(row["branches"]):
                if len(branch)>1:
                    sequences=[self.tokenizer.encode(s,add_special_tokens=False) for s in row["target_texts"][v]]
                    row["semantic_trees"][v]=choice_tree(sequences,self.tokenizer.eos_token_id)
            result.append(row)
        return result
    def values(self,prepared):
        facts=torch.tensor([r["features"]+readiness_features(r) for r in prepared],device=self.device_name,dtype=torch.float32)
        return self.critic(facts).squeeze(-1)
    def verb_distribution(self,prepared,reference=False):
        base=super().verb_distribution(prepared,reference)
        mask=torch.tensor([r["verb_mask"] for r in prepared],device=self.device_name)
        return exploration_distribution(base,mask,self.c.verb_exploration_epsilon)
    def assert_reference_frozen(self):
        if (self.reference_model is None or any(p.requires_grad or p.grad is not None for p in self.reference_model.parameters())
            or self.reference_digest()!=self.anchor_digest):
            raise RuntimeError("Registered supply3 research reference changed")
    def metadata(self):
        result=super().metadata()
        result.update(contract=VERSION,world_version=7,profile="m9",environment_version="m9_regions_v1",
            map_mode=self.c.map_mode,navigation_steps=self.c.navigation_steps,state_variant=self.c.state_variant,
            reward_variant=self.c.reward_variant,gamma_tick=self.c.gamma_tick,verb_exploration_epsilon=self.c.verb_exploration_epsilon,
            critic_features=108,critic_feature_names=result["critic_feature_names"]+list(READINESS_NAMES),
            reference_actor_sha256=self.anchor_digest if self.reference_model is not None else None,
            warm_checkpoint=self.c.warm_checkpoint,release_authorized=False)
        return result

def load(policy,path):
    path=Path(path);manifest=verify_checkpoint(path)
    meta=json.loads((path/"policy.json").read_text())
    current=policy.metadata()
    for key in IDENTITY_KEYS:
        if meta[key]!=current[key]:raise ValueError("M9 checkpoint identity differs: "+key)
    set_peft_model_state_dict(policy.model,load_file(str(path/"adapter/adapter_model.safetensors")))
    policy.critic.load_state_dict(torch.load(path/"critic.pt",map_location=policy.device_name,weights_only=True),strict=True)
    policy.execution_cache.revisions.clear();policy.clear_prefix_cache()
    if policy.digest()!=manifest["actor_sha256"]:raise ValueError("M9 actor digest mismatch")

def save(policy,path,actor_optimizer,critic_optimizer,number,registration_sha):
    path=Path(path)
    if path.exists():raise FileExistsError("Never overwrite checkpoints")
    staging=path.with_name(path.name+".partial")
    staging.mkdir(parents=True,exist_ok=False)
    policy.model.save_pretrained(staging/"adapter",safe_serialization=True)
    torch.save(policy.critic.state_dict(),staging/"critic.pt")
    torch.save(dict(actor=actor_optimizer.state_dict(),critic=critic_optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),staging/"optimizers.pt")
    write(staging/"policy.json",{**policy.metadata(),"update":number,"registration_sha256":registration_sha})
    files={p.relative_to(staging).as_posix():sha(p) for p in staging.rglob("*") if p.is_file()}
    write(staging/"manifest.json",dict(files=files,update=number,actor_sha256=policy.digest(),
        registration_sha256=registration_sha,resume_semantics="No implicit optimizer or episode resume"))
    staging.rename(path)

