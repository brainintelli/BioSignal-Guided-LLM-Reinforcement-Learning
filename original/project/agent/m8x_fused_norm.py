"""Fused frozen-weight Qwen RMSNorm, with only the required input gradient."""
from types import MethodType
import torch
import triton
import triton.language as tl


@triton.jit
def _norm_forward(X,W,Y,R,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);mask=col<D
    x=tl.load(X+row*D+col,mask,other=0).to(tl.float32)
    w=tl.load(W+col,mask,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    tl.store(Y+row*D+col,(x*inv)*w,mask)
    tl.store(R+row,inv)


@triton.jit
def _norm_backward(X,W,R,DY,DX,D:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);mask=col<D
    x=tl.load(X+row*D+col,mask,other=0).to(tl.float32)
    w=tl.load(W+col,mask,other=0).to(tl.float32)
    dy=tl.load(DY+row*D+col,mask,other=0).to(tl.float32)
    inv=tl.load(R+row);norm=x*inv;scaled=dy*w
    dx=inv*(scaled-norm*(tl.sum(scaled*norm,0)/D))
    tl.store(DX+row*D+col,dx,mask)


class FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,eps):
        x=x.contiguous();d=x.shape[-1];rows=x.numel()//d
        y=torch.empty_like(x);r=torch.empty(rows,device=x.device,dtype=torch.float32)
        _norm_forward[(rows,)](x,weight,y,r,d,eps,triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
        if ctx.needs_input_grad[0]:ctx.save_for_backward(x,weight,r)
        return y

    @staticmethod
    def backward(ctx,dy):
        x,weight,r=ctx.saved_tensors;dy=dy.contiguous();dx=torch.empty_like(x);d=x.shape[-1]
        _norm_backward[(x.numel()//d,)](x,weight,r,dy,dx,d,triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
        return dx,None,None


def _forward(module,x):
    weight=module.weight
    revision=(id(weight),weight.data_ptr(),weight._version,weight.dtype,weight.device)
    if getattr(module,'_fused_norm_revision',None)!=revision:
        if weight.requires_grad:raise ValueError('Only frozen base norm weights supported')
        with torch.no_grad():module._unit_offset_weight=(1.+weight.float()).detach()
        module._fused_norm_revision=revision
    return FrozenRMSNorm.apply(x,module._unit_offset_weight,module.eps)


def enable_fused_norm(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    count=0
    for module in model.modules():
        if isinstance(module,Qwen3_5RMSNorm):
            if module.weight.requires_grad:raise ValueError('Norm weight must stay frozen')
            module.forward=MethodType(_forward,module);count+=1
    return count
