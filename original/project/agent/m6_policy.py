"""Hierarchical language-model actor and independent factual-state critic."""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

from agent.m6_contract import CODES, LABELS, VERSION, encode, target_messages, critic_features


class Policy(nn.Module):
    def __init__(self, model_path: str, checkpoint=None, seed=16213, max_tokens=1536,
                 device="cuda", gradient_checkpointing=False):
        super().__init__()
        torch.manual_seed(seed)
        self.device_name, self.max_tokens = device,max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.verb_ids = self._single_ids(CODES)
        self.target_ids = self._single_ids(LABELS)
        cfg = AutoConfig.from_pretrained(model_path,local_files_only=True)
        self.base_config_sha256=hashlib.sha256((Path(model_path)/"config.json").read_bytes()).hexdigest()
        model_class = Qwen3_5ForConditionalGeneration if cfg.model_type == "qwen3_5" else AutoModelForCausalLM
        base = model_class.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
        self.family = cfg.model_type
        self.base_parameters = sum(p.numel() for p in base.parameters())
        suffixes = {"q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","in_proj_b",
                    "in_proj_a","out_proj","gate_proj","up_proj","down_proj"}
        targets = [n for n,m in base.named_modules() if isinstance(m,nn.Linear) and "visual" not in n and n.rsplit(".",1)[-1] in suffixes]
        if checkpoint:
            metadata = json.loads((Path(checkpoint)/"policy.json").read_text())
            if (metadata["contract"] != VERSION or metadata["verb_token_ids"] != self.verb_ids
                or metadata["target_token_ids"] != self.target_ids
                or metadata["base_config_sha256"] != self.base_config_sha256
                or metadata["base_parameters"] != self.base_parameters
                or metadata.get("inference_layout") != "fixed_batch4_left_pad_v1"
                or metadata["max_tokens"] != self.max_tokens):
                raise ValueError("M6 checkpoint/encoding mismatch")
            self.model = PeftModel.from_pretrained(base,str(Path(checkpoint)/"adapter"),is_trainable=True)
        else:
            self.model = get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
                                                       target_modules=targets,bias="none",task_type="CAUSAL_LM"))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            self.model.enable_input_require_grads()
        self.model.to(device)
        self.critic = nn.Sequential(nn.Linear(97,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1)).to(device)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        if checkpoint:
            self.critic.load_state_dict(torch.load(Path(checkpoint)/"critic.pt",map_location=device,weights_only=True))
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
            with self.model.disable_adapter() if reference else nullcontext():
                out=self.model(**batch,use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
                last=out.hidden_states[-1][:,-1,:].float()
                rows=self.model.get_output_embeddings().weight[token_ids].float()
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
            logits=logits.index_copy(0,torch.tensor(active,device=self.device_name),computed)
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

    def digest(self):
        result=hashlib.sha256()
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                result.update(name.encode()); result.update(p.detach().float().cpu().numpy().tobytes())
        return result.hexdigest()

    def metadata(self):
        return {"contract":VERSION,"family":self.family,"base_parameters":self.base_parameters,
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
