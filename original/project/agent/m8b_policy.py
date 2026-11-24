"""Balanced-world actor with the original M6 letter interface and frozen parent.

The fixed-shape forward and probability implementation follows the preserved
M8 implementation. Environment identity and target mapping are independently
bound to the M8B contract; original M8 source files are not modified.
"""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

from agent.m8b_contract import CODES, TARGET_VOCAB, VERSION, encode, target_messages, critic_features, balance_identity
from agent.rl_contract import LABELS
from agent.m8b_policy_identity import PARENT_ACTOR_SHA256, branch_token_indices, validate_checkpoint_metadata, validate_parent_identity


class Policy(nn.Module):
    def __init__(self, model_path: str, checkpoint=None, seed=16213, max_tokens=1536,
                 device="cuda", gradient_checkpointing=False, parent_checkpoint=None, with_reference=True):
        super().__init__()
        if not parent_checkpoint:
            raise ValueError("An explicit released parent checkpoint is mandatory")
        if max_tokens != 1536:
            raise ValueError("M8B requires the registered B4 x 1536 physical shape")
        self.world_identity = balance_identity()
        loading = checkpoint or parent_checkpoint
        torch.manual_seed(seed)
        self.device_name, self.max_tokens = device,max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.verb_ids = self._single_ids(CODES)
        self.target_ids = self._single_ids(TARGET_VOCAB)
        cfg = AutoConfig.from_pretrained(model_path,local_files_only=True)
        self.base_config_sha256=hashlib.sha256((Path(model_path)/"config.json").read_bytes()).hexdigest()
        model_class = Qwen3_5ForConditionalGeneration if cfg.model_type == "qwen3_5" else AutoModelForCausalLM
        base = model_class.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
        self.family = cfg.model_type
        self.base_parameters = sum(p.numel() for p in base.parameters())
        metadata = json.loads((Path(loading)/"policy.json").read_text())
        validate_checkpoint_metadata(metadata,self.verb_ids,self.target_ids,self.base_config_sha256,
                                     self.base_parameters,max_tokens,checkpoint is None)
        self.model = PeftModel.from_pretrained(base,str(Path(loading)/"adapter"),is_trainable=True)
        # Explicit stable parameter names survive freezing; never hash an empty requires_grad subset.
        self.actor_parameter_names = tuple(n for n,p in self.model.named_parameters() if p.requires_grad)
        if not self.actor_parameter_names or any("lora_" not in n for n in self.actor_parameter_names):
            raise ValueError("Actor optimizer must contain only explicit LoRA parameters")
        self.reference_model = None
        if with_reference:
            parent_meta = json.loads((Path(parent_checkpoint)/"policy.json").read_text())
            validate_checkpoint_metadata(parent_meta,self.verb_ids,self.target_ids,self.base_config_sha256,
                                         self.base_parameters,max_tokens,True)
            reference_base = model_class.from_pretrained(model_path,local_files_only=True,
                                                        dtype=torch.bfloat16,attn_implementation="sdpa")
            self.reference_model = PeftModel.from_pretrained(reference_base,
                str(Path(parent_checkpoint)/"adapter"),is_trainable=False).to(device)
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
            if self.reference_digest() != PARENT_ACTOR_SHA256:
                raise ValueError("Reference is not the released parent actor")
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            self.model.enable_input_require_grads()
        self.model.to(device)
        self.critic = nn.Sequential(nn.Linear(97,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1)).to(device)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        self.critic.load_state_dict(torch.load(Path(loading)/"critic.pt",map_location=device,weights_only=True),strict=True)
        if checkpoint:
            if self.digest() != metadata.get("actor_sha256"):
                raise ValueError("Saved M8B actor digest mismatch")
        elif self.digest() != PARENT_ACTOR_SHA256:
            raise ValueError("Actor initialization is not the released M6 parent")
        if self.reference_model is not None and not checkpoint:
            validate_parent_identity(self.digest(),self.reference_digest())
        self.eval()

    def _single_ids(self, words):
        result = [self.tokenizer.encode(word,add_special_tokens=False) for word in words]
        if any(len(x)!=1 for x in result) or len({x[0] for x in result}) != len(result):
            raise ValueError("Choice codes must be distinct single tokens")
        return [x[0] for x in result]

    def _tokens(self,messages):
        text = self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids = self.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)>self.max_tokens:
            raise ValueError(f"Observation {len(ids)} exceeds {self.max_tokens}; no silent truncation")
        return ids

    def prepare(self,observations,salt="m6"):
        result=[]
        for obs in observations:
            row=encode(obs,salt)
            row.update(input_ids=self._tokens(row["messages"]),features=critic_features(row),target_input_ids={})
            result.append(row)
        return result

    def _target_tokens(self,row,verb):
        if verb not in row["target_input_ids"]:
            row["target_input_ids"][verb]=self._tokens(target_messages(row,verb))
        return row["target_input_ids"][verb]

    def _logits(self,inputs,token_ids,reference=False):
        # Both axes must be fixed: BF16 kernels differed when either padded
        # sequence length or the number of active target branches changed.
        # Extra rows have no action, reward or loss; outputs are sliced away.
        if not inputs or any(len(ids)>self.max_tokens for ids in inputs):
            raise ValueError("Empty input or context overflow; never truncate legal choices")
        chunks=[]
        for start in range(0,len(inputs),4):
            real=inputs[start:start+4]
            physical=real+[real[-1]]*(4-len(real))
            batch=self.tokenizer.pad({"input_ids":physical},padding="max_length",max_length=self.max_tokens,return_tensors="pt")
            batch={k:v.to(self.device_name) for k,v in batch.items()}
            selected_model = self.reference_model if reference else self.model
            if selected_model is None:
                raise RuntimeError("Reference requested from an inference-only policy")
            with torch.no_grad() if reference else nullcontext():
                out=selected_model(**batch,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
                last=out.hidden_states[-1][:,-1,:].float()
                rows=selected_model.get_output_embeddings().weight[token_ids].float()
                chunks.append(torch.nn.functional.linear(last,rows)[:len(real)])
        return torch.cat(chunks,dim=0)

    def verb_distribution(self,prepared,reference=False):
        logits=self._logits([r["input_ids"] for r in prepared],self.verb_ids,reference)
        mask=torch.tensor([r["verb_mask"] for r in prepared],device=self.device_name,dtype=torch.bool)
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-1.e9))

    def target_distribution(self,prepared,verbs,reference=False):
        # A single legal target requires no invented model call or policy choice.
        logits=torch.zeros((len(prepared),len(LABELS)),device=self.device_name)
        active=[i for i,(r,v) in enumerate(zip(prepared,verbs)) if len(r["branches"][int(v)])>1]
        if active:
            computed=self._logits([self._target_tokens(prepared[i],int(verbs[i])) for i in active],self.target_ids,reference)
            # M8B preserves the original 64-letter branch-position mapping.
            mapped=[]
            for j,i in enumerate(active):
                indices=branch_token_indices(prepared[i],int(verbs[i]),len(self.target_ids))
                picked=computed[j,torch.tensor(indices,device=self.device_name)]
                mapped.append(torch.nn.functional.pad(picked,(0,len(LABELS)-len(indices))))
            logits=logits.index_copy(0,torch.tensor(active,device=self.device_name),torch.stack(mapped))
        lengths=torch.tensor([len(r["branches"][int(v)]) for r,v in zip(prepared,verbs)],device=self.device_name)
        mask=torch.arange(len(LABELS),device=self.device_name)[None,:] >= lengths[:,None]
        return torch.distributions.Categorical(logits=logits.masked_fill(mask,-1.e9))

    def values(self,prepared):
        facts=torch.tensor([r["features"] for r in prepared],device=self.device_name,dtype=torch.float32)
        return self.critic(facts).squeeze(-1)

    def distributions(self,prepared,verbs,reference=False):
        return self.verb_distribution(prepared,reference),self.target_distribution(prepared,verbs,reference)

    @torch.no_grad()
    def act(self,prepared,greedy=False,with_reference=False):
        vd=self.verb_distribution(prepared)
        verbs=vd.probs.argmax(-1) if greedy else vd.sample()
        td=self.target_distribution(prepared,verbs.tolist())
        targets=td.probs.argmax(-1) if greedy else td.sample()
        refv,reft=self.distributions(prepared,verbs.tolist(),True) if with_reference else (None,None)
        return {"verbs":verbs.tolist(),"targets":targets.tolist(),"values":self.values(prepared).tolist(),
                "logp":(vd.log_prob(verbs)+td.log_prob(targets)).tolist(),"verb_logp":vd.log_prob(verbs).tolist(),
                "reference_verb":refv.probs.tolist() if refv is not None else None,
                "reference_target":reft.probs.tolist() if reft is not None else None,
                "actions":[r["branches"][v][t] for r,v,t in zip(prepared,verbs.tolist(),targets.tolist())]}

    def _digest_model(self,model):
        result=hashlib.sha256()
        parameters=dict(model.named_parameters())
        for name in self.actor_parameter_names:
            if name not in parameters: raise ValueError("Reference/actor LoRA parameter names differ")
            result.update(name.encode())
            result.update(parameters[name].detach().float().cpu().numpy().tobytes())
        return result.hexdigest()

    def digest(self):
        return self._digest_model(self.model)

    def reference_digest(self):
        if self.reference_model is None: raise RuntimeError("No reference model loaded")
        return self._digest_model(self.reference_model)

    def assert_reference_frozen(self):
        if self.reference_model is None or any(p.requires_grad or p.grad is not None for p in self.reference_model.parameters()):
            raise RuntimeError("Reference is not completely frozen")
        if self.reference_digest()!=PARENT_ACTOR_SHA256:
            raise RuntimeError("Frozen parent reference changed")

    def metadata(self):
        current_world_identity = balance_identity()
        if current_world_identity != self.world_identity:
            raise RuntimeError("Balanced rules changed after policy construction")
        return {"contract":VERSION,**self.world_identity,"actor_sha256":self.digest(),"reference_actor_sha256":PARENT_ACTOR_SHA256,"reference_implementation":"independent_frozen_parent", "critic_features":97,"family":self.family,"base_parameters":self.base_parameters,
                "base_config_sha256":self.base_config_sha256,
                "actor_trainable_parameters":sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "critic_parameters":sum(p.numel() for p in self.critic.parameters()),
                "verb_token_ids":self.verb_ids,"target_token_ids":self.target_ids,"max_tokens":self.max_tokens,
                "inference_layout":"fixed_batch4_left_pad_v1","physical_forward_shape":[4,self.max_tokens],
                "padding_rows_are_environment_decisions":False,
                "critic_input":"97 features derived solely from serialized actor facts",
                "action_probability":"P(verb|observation)*P(target|observation,verb)",
                "vision_used":False,"task_goals":False,"policy_override":False}

    def save(self,path):
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path/"adapter",safe_serialization=True)
        torch.save(self.critic.state_dict(),path/"critic.pt")
        (path/"policy.json").write_text(json.dumps(self.metadata(),indent=2),encoding="utf-8")
