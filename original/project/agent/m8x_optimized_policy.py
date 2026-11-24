"""Registered unpadded KV execution with versioned projections and fused norms."""
from agent.m8x_causal_policy import CausalKVPolicy
from agent.m8x_merged_lora import enable_merged_lora
from agent.m8x_fused_norm import enable_fused_norm


class OptimizedPolicy(CausalKVPolicy):
    def __init__(self,*args,registration=None,fused_norm=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.registration_identity=registration['registration_sha256'] if registration else None
        self.optimized_projection_counts={'actor':enable_merged_lora(self.model)}
        self.optimized_norm_counts={'actor':enable_fused_norm(self.model) if fused_norm else 0}
        if self.reference_model is not None:
            self.optimized_projection_counts['reference']=enable_merged_lora(self.reference_model)
            self.optimized_norm_counts['reference']=enable_fused_norm(self.reference_model) if fused_norm else 0

    def metadata(self):
        result=super().metadata()
        result.update(inference_layout='unpadded_full_verb_kv_merged_v1',
            optimization_registered=self.registration_identity is not None,
            experiment_registration_sha256=self.registration_identity,
            projection_arithmetic='versioned_FP32_merge_BF16_projection_FP32_LoRA_gradients',
            optimized_projections=self.optimized_projection_counts,
            fused_zero_centered_norms=self.optimized_norm_counts,
            formal_training_started=False,release_authorized=False)
        return result
