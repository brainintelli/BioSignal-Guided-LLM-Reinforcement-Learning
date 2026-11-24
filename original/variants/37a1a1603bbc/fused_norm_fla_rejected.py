"""Use installed FLA RMSNorm for frozen zero-centered Qwen norm weights."""
from types import MethodType
import torch


def _forward(module,x):
    from fla.modules.layernorm import rms_norm
    weight=module.weight
    revision=(id(weight),weight.data_ptr(),weight._version,weight.dtype,weight.device)
    if getattr(module,'_fused_norm_revision',None)!=revision:
        if weight.requires_grad:raise ValueError('Only frozen base norm weights supported')
        with torch.no_grad():module._unit_offset_weight=(1.+weight.float()).detach()
        module._fused_norm_revision=revision
    return rms_norm(x,module._unit_offset_weight,None,eps=module.eps)


def enable_fused_norm(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    count=0
    for module in model.modules():
        if isinstance(module,Qwen3_5RMSNorm):
            if module.weight.requires_grad:raise ValueError('Norm weight must stay frozen')
            module.forward=MethodType(_forward,module);count+=1
    return count
