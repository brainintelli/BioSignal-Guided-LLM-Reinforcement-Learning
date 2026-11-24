"""Actor-only migration, static KV, and opt-in deterministic scene caching.

Old critic and optimizer state are never opened. Every cold/warm, reference and
autograd execution uses the same explicit system/scene/dynamic boundaries.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

from agent.m8s2_policy import BatchedSemanticPolicy as PreviousPolicy
from agent.m8s2_cache import fresh_cache
from agent.m8s2_kernels import pin_kernels
from agent.m8s_trie import choice_tree
from agent.m8x_kv_policy import branch_cache
from agent.m8x_runtime import kernel_identity
from agent.m8x_policy_identity import validate_checkpoint_metadata, PARENT_ACTOR_SHA256
from agent.m8x_contract import CODES as OLD_CODES, TARGET_VOCAB
from agent.m8x_merged_lora import enable_merged_lora
from agent.m8x_fused_norm import enable_fused_norm
from agent.m8x_fused_mlp import enable_fused_mlp
from agent.m8x_grouped_lora import enable_grouped_lora, ExecutionCache, _prepared_linear, _prepared_norm
from agent.m8s3_contract import (encode, render_scene, CODES, RULES, STATE_HEADER, VERSION,
                                FEATURE_DIM, FEATURE_NAMES, REWARD_CONFIG, balance_identity)


class BatchedSemanticPolicy(PreviousPolicy):
    def __init__(self, model_path, checkpoint=None, seed=16213, max_tokens=1536,
                 device="cuda", gradient_checkpointing=False, parent_checkpoint=None,
                 with_reference=True, reset_critic=False, scene_cache_size=8,
                 prefix_layout="system_only"):
        # Deliberately do not invoke old __init__: it opens an incompatible critic.
        nn.Module.__init__(self)
        if not reset_critic:
            raise ValueError("New 96-feature critic requires explicit reset_critic=True; old critic is incompatible")
        if not parent_checkpoint or not checkpoint:
            raise ValueError("Explicit actor source and frozen parent checkpoint paths are required")
        if max_tokens != 1536 or gradient_checkpointing:
            raise ValueError("This diagnostic preserves max_tokens=1536 and cache-compatible autograd")
        if prefix_layout not in ("system_scene", "system_only"):
            raise ValueError("Unknown canonical prefix layout")
        if type(scene_cache_size) is not int or not 0 <= scene_cache_size <= 32:
            raise ValueError("Scene LRU capacity must be 0..32")
        self.pinned_kernels = pin_kernels()
        self.world_identity, self.kernel_identity = balance_identity(), kernel_identity()
        self.device_name, self.max_tokens = device, max_tokens
        self.prefix_layout, self.scene_cache_size = prefix_layout, scene_cache_size
        self.actor_source = str(Path(checkpoint))
        self.critic_initialization = "fresh_96_features_old_critic_not_read"
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        old_verbs, old_targets = self._single_ids(OLD_CODES), self._single_ids(TARGET_VOCAB)
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if cfg.model_type != "qwen3_5":
            raise ValueError("Only the validated Qwen3.5 backbone is supported")
        self.family = cfg.model_type
        self.base_config_sha256 = hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()
        base = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.base_parameters = sum(p.numel() for p in base.parameters())
        source = json.loads((Path(checkpoint) / "policy.json").read_text())
        validate_checkpoint_metadata(source, old_verbs, old_targets, self.base_config_sha256,
                                     self.base_parameters, max_tokens, False, self.kernel_identity)
        self.model = PeftModel.from_pretrained(base, str(Path(checkpoint) / "adapter"), is_trainable=True).to(device)
        self.actor_parameter_names = tuple(n for n, p in self.model.named_parameters() if p.requires_grad)
        if not self.actor_parameter_names or any("lora_" not in n for n in self.actor_parameter_names):
            raise ValueError("Actor must expose exactly its LoRA trainable parameter set")
        if self.digest() != source["actor_sha256"]:
            raise ValueError("Source actor checkpoint digest mismatch")
        self.reference_model = None
        if with_reference:
            parent_meta = json.loads((Path(parent_checkpoint) / "policy.json").read_text())
            validate_checkpoint_metadata(parent_meta, old_verbs, old_targets, self.base_config_sha256,
                                         self.base_parameters, max_tokens, True, self.kernel_identity)
            reference_base = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
            self.reference_model = PeftModel.from_pretrained(
                reference_base, str(Path(parent_checkpoint) / "adapter"), is_trainable=False).to(device)
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
            if self.reference_digest() != PARENT_ACTOR_SHA256:
                raise ValueError("Frozen reference is not the released parent actor")
        self.critic = nn.Sequential(nn.Linear(FEATURE_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)).to(device)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        self.verb_ids = self._single_ids(CODES)
        self.direction_ids = self._single_ids(("向上", "向右", "向下", "向左"))
        self.target_ids = []
        self.optimization, self._kv_scope = "compact", None
        self.semantic_node_calls = 0
        self._enable_execution()
        template = self.tokenizer.apply_chat_template(
            [dict(role="system", content=RULES), dict(role="user", content=STATE_HEADER + "boundary")],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        marker = "<|im_start|>user\n"
        if template.count(marker) != 1:
            raise ValueError("Unexpected chat-template boundary")
        self.system_ids = self.tokenizer.encode(template.split(marker)[0] + marker + STATE_HEADER, add_special_tokens=False)
        self.prefix_ids = self.system_ids
        self._persistent_prefixes = {}
        self._persistent_scenes = {False: OrderedDict(), True: OrderedDict()}
        self._scene_revisions = {}
        self.cache_stats = dict(system_hits=0, system_misses=0, state_hits=0, state_misses=0,
                                scene_hits=0, scene_misses=0, scene_persistent_hits=0, scene_reused_tokens=0,
                                branch_batches=0, branch_rows=0, forward_tokens=0, forward_calls=0)
        self.eval()

    def _enable_execution(self):
        from peft.tuners.lora.layer import Linear
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        self.execution_cache, self.execution_parts = ExecutionCache(), {}
        self.optimized_projection_counts, self.optimized_norm_counts = {}, {}
        self.group_counts, self.fused_mlp_counts = {}, {}
        for reference, model in ((False, self.model), (True, self.reference_model)):
            if model is None:
                continue
            name = "reference" if reference else "actor"
            self.optimized_projection_counts[name] = enable_merged_lora(model)
            self.optimized_norm_counts[name] = enable_fused_norm(model)
            self.group_counts[name] = enable_grouped_lora(model)
            self.fused_mlp_counts[name] = enable_fused_mlp(model)
            groups = [m._grouped_projection for m in model.modules() if hasattr(m, "_grouped_projection")]
            grouped = {id(layer) for group in groups for layer in group.layers}
            linears = [m for m in model.modules() if isinstance(m, Linear) and id(m) not in grouped]
            norms = [m for m in model.modules() if isinstance(m, Qwen3_5RMSNorm)]
            for group in groups:
                group.group.managed = True
            for module in linears:
                module.forward = MethodType(_prepared_linear, module)
            for module in norms:
                module.forward = MethodType(_prepared_norm, module)
            self.execution_parts[reference] = groups, linears, norms

    def prepare(self, observations, salt="m6"):
        result = []
        for obs in observations:
            row = encode(obs, salt)
            ids = self._tokens(row["messages"])
            text = self.tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            marker = "<|im_start|>user\n"
            boundary = text.index(marker) + len(marker) + len(render_scene(row["semantic_state"]))
            scene = self.tokenizer.encode(text[:boundary], add_special_tokens=False)
            if ids[:len(self.system_ids)] != self.system_ids or ids[:len(scene)] != scene:
                raise ValueError("Tokenizer merged across a declared prefix boundary")
            if not len(self.system_ids) < len(scene) < len(ids):
                raise ValueError("Missing canonical scene/dynamic token partition")
            row.update(input_ids=ids, kv_prefix_ids=ids, scene_prefix_ids=scene,
                       normalized_targets={}, semantic_trees={})
            for v, branch in enumerate(row["branches"]):
                if len(branch) > 1:
                    sequences = [self.tokenizer.encode(s, add_special_tokens=False) for s in row["target_texts"][v]]
                    row["semantic_trees"][v] = choice_tree(sequences, self.tokenizer.eos_token_id)
            result.append(row)
        return result

    @contextmanager
    def _scope(self, prepared, reference):
        with super()._scope(prepared, reference):
            declarations = self._kv_scope.setdefault("scene_declarations", {})
            for row in prepared:
                declarations[tuple(row["input_ids"])] = tuple(row["scene_prefix_ids"])
            yield

    def clear_prefix_cache(self):
        self._persistent_prefixes.clear()
        for scenes in self._persistent_scenes.values():
            scenes.clear()
        self._scene_revisions.clear()

    def _root(self, ids, reference):
        if self.prefix_layout == "system_only":
            return super()._root(ids, reference)
        scope, key = self._kv_scope, tuple(ids)
        if key not in scope["roots"]:
            raise ValueError("Unregistered state root")
        if scope["roots"][key] is not None:
            self.cache_stats["state_hits"] += 1
            return scope["roots"][key]
        self.cache_stats["state_misses"] += 1
        model = self.reference_model if reference else self.model
        revision = self.execution_cache.revisions[reference]
        static = scope.get("static_prefix")
        if static is None:
            previous = self._persistent_prefixes.get(reference) if not torch.is_grad_enabled() else None
            if previous is not None and previous[0] == revision:
                static = previous[1]
                self.cache_stats["system_hits"] += 1
            else:
                static = fresh_cache(model.get_base_model().config)
                self._forward_batch(model, [self.system_ids], static)
                self.cache_stats["system_misses"] += 1
                if not torch.is_grad_enabled():
                    self._persistent_prefixes[reference] = revision, static
            scope["static_prefix"] = static
        else:
            self.cache_stats["system_hits"] += 1
        scene_ids = scope["scene_declarations"][key]
        local_scenes = scope.setdefault("scene_roots", {})
        persistent = self._persistent_scenes[reference]
        if self._scene_revisions.get(reference) != revision:
            persistent.clear()
            self._scene_revisions[reference] = revision
        scene = local_scenes.get(scene_ids)
        if scene is None and not torch.is_grad_enabled() and scene_ids in persistent:
            scene = persistent.pop(scene_ids)
            persistent[scene_ids] = scene
            self.cache_stats["scene_persistent_hits"] += 1
        if scene is None:
            scene = branch_cache(static)
            self._forward_batch(model, [list(scene_ids[len(self.system_ids):])], scene)
            self.cache_stats["scene_misses"] += 1
            if not torch.is_grad_enabled() and self.scene_cache_size:
                persistent[scene_ids] = scene
                while len(persistent) > self.scene_cache_size:
                    persistent.popitem(last=False)
        else:
            self.cache_stats["scene_hits"] += 1
            self.cache_stats["scene_reused_tokens"] += len(scene_ids) - len(self.system_ids)
        local_scenes[scene_ids] = scene
        cache = branch_cache(scene)
        last = self._forward_batch(model, [ids[len(scene_ids):]], cache)
        scope["roots"][key] = cache, last
        return cache, last

    def load_critic_state(self, state_dict, metadata):
        if (metadata.get("contract") != VERSION or metadata.get("critic_features") != FEATURE_DIM
                or metadata.get("critic_feature_names") != list(FEATURE_NAMES)
                or any(metadata.get(k) != v for k, v in self.world_identity.items())):
            raise ValueError("Critic feature/world identity mismatch; old 97-feature critic is rejected")
        if tuple(state_dict["0.weight"].shape) != (128, FEATURE_DIM):
            raise ValueError("Critic first-layer shape mismatch")
        self.critic.load_state_dict(state_dict, strict=True)
        self.critic_initialization = "explicit_new_world_96_feature_state"

    def metadata(self):
        if balance_identity() != self.world_identity or kernel_identity() != self.kernel_identity:
            raise RuntimeError("World or kernel identity changed since construction")
        return dict(contract=VERSION, **self.world_identity, kernel_identity=deepcopy(self.kernel_identity),
                    reward_config=deepcopy(REWARD_CONFIG), actor_sha256=self.digest(), actor_source=self.actor_source,
                    reference_actor_sha256=PARENT_ACTOR_SHA256 if self.reference_model is not None else None,
                    critic_features=FEATURE_DIM, critic_feature_names=list(FEATURE_NAMES), critic_initialization=self.critic_initialization,
                    family=self.family, base_parameters=self.base_parameters, base_config_sha256=self.base_config_sha256,
                    verb_token_ids=self.verb_ids, target_token_ids="complete_semantic_names", direction_token_ids=self.direction_ids,
                    max_tokens=self.max_tokens, inference_layout="semantic_v3_" + self.prefix_layout,
                    default_prefix_layout="system_only",
                    default_layout_reason="Node5 paired forward probe: extra scene forward overhead outweighed token savings; scene reuse remains opt-in",
                    system_prefix_tokens=len(self.system_ids), scene_cache_size=self.scene_cache_size,
                    scene_cache_active=self.prefix_layout == "system_scene",
                    cache_lifetime=("exact-token scene LRU, parameter-versioned no-grad; gradient caches graph-local"
                                    if self.prefix_layout == "system_scene" else
                                    "fixed system prefix, parameter-versioned no-grad; gradient caches graph-local"),
                    pinned_fla_configs=self.pinned_kernels, grouped_projections=self.group_counts,
                    action_probability="semantic verb probability times full semantic target-trie probability; shared state root",
                    formal_training_started=False, checkpoint_writes=0, release_authorized=False)


Policy = BatchedSemanticPolicy
