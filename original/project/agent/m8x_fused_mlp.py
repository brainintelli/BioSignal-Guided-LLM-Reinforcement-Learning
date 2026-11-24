"""Fused SiLU gating with native intermediate BF16 rounding preserved."""
from types import MethodType
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X,U,Y,N,D:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=index<N
    offset=(index//D)*S+index%D
    x=tl.load(X+offset,mask,other=0).to(tl.float32)
    u=tl.load(U+offset,mask,other=0).to(tl.float32)
    s=(x/(1.+tl.exp(-x))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y+index,s*u,mask)


@triton.jit
def _backward(X,U,G,DX,DU,N,D:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=index<N
    offset=(index//D)*S+index%D
    x=tl.load(X+offset,mask,other=0).to(tl.float32)
    u=tl.load(U+offset,mask,other=0).to(tl.float32)
    g=tl.load(G+index,mask,other=0).to(tl.float32)
    sigmoid=1./(1.+tl.exp(-x))
    s=(x*sigmoid).to(X.dtype.element_ty).to(tl.float32)
    intermediate=(g*u).to(X.dtype.element_ty).to(tl.float32)
    dx=intermediate*(sigmoid*(1.+x*(1.-sigmoid)))
    tl.store(DX+index,dx,mask);tl.store(DU+index,g*s,mask)


class SiluGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,u):
        if x.shape!=u.shape or x.stride()!=u.stride() or x.stride(-1)!=1:raise ValueError('Incompatible gating views')
        d=x.shape[-1];stride=x.stride(-2);n=x.numel()
        y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        _forward[(triton.cdiv(n,256),)](x,u,y,n,d,stride,256,enable_fp_fusion=False)
        if any(ctx.needs_input_grad):ctx.save_for_backward(x,u)
        return y

    @staticmethod
    def backward(ctx,g):
        x,u=ctx.saved_tensors;g=g.contiguous()
        dx=torch.empty_like(g);du=torch.empty_like(g);n=x.numel()
        _backward[(triton.cdiv(n,256),)](x,u,g,dx,du,n,x.shape[-1],x.stride(-2),256,enable_fp_fusion=False)
        return dx,du


def _mlp(module,x):return module.down_proj(SiluGate.apply(module.gate_proj(x),module.up_proj(x)))


def enable_fused_mlp(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
    count=0
    for module in model.modules():
        if isinstance(module,Qwen3_5MLP):module.forward=MethodType(_mlp,module);count+=1
    return count
