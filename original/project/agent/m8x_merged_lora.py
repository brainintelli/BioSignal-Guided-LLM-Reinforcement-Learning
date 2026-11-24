"""Cache the effective projection per optimizer version, retain LoRA gradients.

Base parameters are never modified. The effective projection is rounded to the
base dtype once per version; this is a new explicitly registered numeric layout.
Only ordinary bias-free/dropout-free LoRA linear adapters are supported.
"""
from types import MethodType
import torch
from torch.nn import functional as F


class MergedWeight:
    def __init__(self):self.revision=None;self.weight=None

    @torch.no_grad()
    def get(self,base,a,b,scale):
        revision=tuple((id(p),p.data_ptr(),p._version,p.dtype,p.device) for p in (base,a,b))+(scale,)
        if revision!=self.revision:
            dtype=torch.float64 if base.dtype==torch.float64 else torch.float32
            self.weight=(base.to(dtype)+(b.to(dtype)@a.to(dtype))*scale).to(base.dtype)
            self.revision=revision
        return self.weight


class MergedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,a,b,weight,bias,scale):
        ctx.scale=scale
        if any(ctx.needs_input_grad[:3]):ctx.save_for_backward(x,a,b,weight)
        return F.linear(x,weight,bias)

    @staticmethod
    def backward(ctx,grad):
        x,a,b,weight=ctx.saved_tensors
        dx=da=db=None
        if ctx.needs_input_grad[0]:dx=F.linear(grad,weight.t())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
            xx=x.reshape(-1,x.shape[-1]).to(dtype)
            gg=grad.reshape(-1,grad.shape[-1]).to(dtype)
            if ctx.needs_input_grad[1]:da=(((gg@b.to(dtype)).t()@xx)*ctx.scale).to(a.dtype)
            if ctx.needs_input_grad[2]:db=((gg.t()@(xx@a.to(dtype).t()))*ctx.scale).to(b.dtype)
        return dx,da,db,None,None,None


def _forward(module,x,*args,**kwargs):
    if args or kwargs or module.disable_adapters or module.merged:
        raise ValueError('Unsupported merged-cache adapter invocation')
    name=module.active_adapters[0]
    a=module.lora_A[name].weight;b=module.lora_B[name].weight
    scale=module.scaling[name]
    base=module.get_base_layer()
    weight=module._versioned_merged_weight.get(base.weight,a,b,scale)
    return MergedLinear.apply(x,a,b,weight,base.bias,scale)


def enable_merged_lora(model):
    from peft.tuners.lora.layer import Linear
    count=0
    for module in model.modules():
        if not isinstance(module,Linear):continue
        names=module.active_adapters
        if len(names)!=1:raise ValueError('Exactly one LoRA adapter required')
        name=names[0];base=module.get_base_layer()
        if (module.fan_in_fan_out or module.merged or module.disable_adapters
                or module.use_dora.get(name,False) or getattr(module,'lora_variant',{})
                or getattr(module.lora_dropout[name],'p',0)!=0 or base.weight.requires_grad
                or module.lora_A[name].bias is not None or module.lora_B[name].bias is not None
                or (base.bias is not None and base.bias.requires_grad)):
            raise ValueError('Adapter is not a frozen-base, ordinary dropout-free LoRA projection')
        module._versioned_merged_weight=MergedWeight()
        module.forward=MethodType(_forward,module)
        count+=1
    if count==0:raise ValueError('No LoRA projections found')
    return count
