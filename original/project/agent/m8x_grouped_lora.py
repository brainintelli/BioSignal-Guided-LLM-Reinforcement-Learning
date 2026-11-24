"""Group parallel LoRA projections while preserving independent A/B gradients."""
from types import MethodType
from contextlib import contextmanager
import torch
from torch.nn import functional as F


class GroupedProjection:
    def __init__(self,rows):
        self.rows=rows;self.revision=None;self.managed=False
        self.widths=[w.shape[0] for w,a,b,s in rows]
        self.ranks=[a.shape[0] for w,a,b,s in rows]

    @torch.no_grad()
    def refresh(self):
        revision=tuple((id(p),p.data_ptr(),p._version,p.dtype,p.device) for row in self.rows for p in row[:3])
        if revision==self.revision:return
        dtype=torch.float64 if self.rows[0][0].dtype==torch.float64 else torch.float32
        self.weight=torch.cat([(w.to(dtype)+(b.to(dtype)@a.to(dtype))*s).to(w.dtype) for w,a,b,s in self.rows])
        self.a=torch.cat([a.to(dtype) for w,a,b,s in self.rows])
        self.b=torch.block_diag(*[b.to(dtype)*s for w,a,b,s in self.rows])
        self.revision=revision

    def __call__(self,x):
        if not self.managed:self.refresh()
        parameters=[p for row in self.rows for p in row[1:3]]
        return GroupedLinear.apply(x,self,*parameters).split(self.widths,dim=-1)


class GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group,*parameters):
        ctx.group=group
        if any(ctx.needs_input_grad):ctx.save_for_backward(x,group.weight,group.a,group.b)
        return F.linear(x,group.weight)

    @staticmethod
    def backward(ctx,grad):
        x,weight,a,b=ctx.saved_tensors;group=ctx.group
        dx=F.linear(grad,weight.t()) if ctx.needs_input_grad[0] else None
        dtype=a.dtype;xx=x.reshape(-1,x.shape[-1]).to(dtype);gg=grad.reshape(-1,grad.shape[-1]).to(dtype)
        da=(gg@b).t()@xx
        db=gg.t()@(xx@a.t())
        gradients=[];out=rank=0
        for w,pa,pb,scale in group.rows:
            size=w.shape[0];r=pa.shape[0]
            gradients.extend([da[rank:rank+r].to(pa.dtype),
                              (db[out:out+size,rank:rank+r]*scale).to(pb.dtype)])
            out+=size;rank+=r
        return dx,None,*gradients


class ProjectionSequence:
    def __init__(self,layers):
        rows=[]
        for layer in layers:
            name=layer.active_adapters[0];base=layer.get_base_layer()
            if base.bias is not None:raise ValueError('Grouped projection requires bias-free layers')
            rows.append((base.weight,layer.lora_A[name].weight,layer.lora_B[name].weight,layer.scaling[name]))
        self.group=GroupedProjection(rows);self.layers=layers
        self.pending=None

    def call(self,index,x):
        if index==0:
            if self.pending is not None:raise RuntimeError('Unconsumed parallel projection')
            self.pending=[x,self.group(x),1]
            return self.pending[1][0]
        if self.pending is None or self.pending[0] is not x or self.pending[2]!=index:
            raise RuntimeError('Unexpected projection execution order')
        result=self.pending[1][index];self.pending[2]+=1
        if index==len(self.layers)-1:self.pending=None
        return result


def enable_grouped_lora(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP,Qwen3_5GatedDeltaNet,Qwen3_5Attention
    counts={'groups':0,'projections':0}
    for module in list(model.modules()):
        if isinstance(module,Qwen3_5MLP):names=('gate_proj','up_proj')
        elif isinstance(module,Qwen3_5GatedDeltaNet):names=('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')
        elif isinstance(module,Qwen3_5Attention):names=('q_proj','k_proj','v_proj')
        else:continue
        layers=[getattr(module,name) for name in names]
        sequence=ProjectionSequence(layers)
        module._grouped_projection=sequence
        for index,layer in enumerate(layers):
            def forward(layer,x,*args,_index=index,_sequence=sequence,**kwargs):
                if args or kwargs or layer.disable_adapters or layer.merged:raise ValueError('Unsupported grouped invocation')
                return _sequence.call(_index,x)
            layer.forward=MethodType(forward,layer)
        counts['groups']+=1;counts['projections']+=len(layers)
    if not counts['groups']:raise ValueError('No parallel projections found')
    return counts


from agent.m8x_optimized_policy import OptimizedPolicy
from agent.m8x_fast_policy import parameter_revision
from agent.m8x_merged_lora import MergedLinear
from agent.m8x_fused_norm import FrozenRMSNorm


class ExecutionCache:
    def __init__(self):self.revisions={}

    def prepare(self,reference,revision,build):
        if self.revisions.get(reference)!=revision:
            build()
            self.revisions[reference]=revision


def _prepared_linear(module,x,*args,**kwargs):
    if args or kwargs or module.disable_adapters or module.merged:raise ValueError('Unsupported projection invocation')
    name=module.active_adapters[0]
    return MergedLinear.apply(x,module.lora_A[name].weight,module.lora_B[name].weight,
                              module._versioned_merged_weight.weight,module.get_base_layer().bias,module.scaling[name])


def _prepared_norm(module,x):return FrozenRMSNorm.apply(x,module._unit_offset_weight,module.eps)


class GroupedPolicy(OptimizedPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.group_counts={'actor':enable_grouped_lora(self.model)}
        if self.reference_model is not None:self.group_counts['reference']=enable_grouped_lora(self.reference_model)
        from agent.m8x_fused_mlp import enable_fused_mlp
        self.fused_mlp_counts={'actor':enable_fused_mlp(self.model)}
        if self.reference_model is not None:self.fused_mlp_counts['reference']=enable_fused_mlp(self.reference_model)
        self.execution_cache=ExecutionCache();self.execution_parts={}
        from peft.tuners.lora.layer import Linear
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        for reference,model in ((False,self.model),(True,self.reference_model)):
            if model is None:continue
            groups=[m._grouped_projection for m in model.modules() if hasattr(m,'_grouped_projection')]
            grouped={id(layer) for group in groups for layer in group.layers}
            linears=[m for m in model.modules() if isinstance(m,Linear) and id(m) not in grouped]
            norms=[m for m in model.modules() if isinstance(m,Qwen3_5RMSNorm)]
            for group in groups:group.group.managed=True
            for module in linears:module.forward=MethodType(_prepared_linear,module)
            for module in norms:module.forward=MethodType(_prepared_norm,module)
            self.execution_parts[reference]=(groups,linears,norms)

    @torch.no_grad()
    def _prepare_execution(self,reference):
        model=self.reference_model if reference else self.model
        revision=parameter_revision(model)
        def build():
            groups,linears,norms=self.execution_parts[reference]
            for sequence in groups:
                if sequence.pending is not None:raise RuntimeError('Projection branch leaked outside its call')
                rows=[]
                for layer in sequence.layers:
                    name=layer.active_adapters[0]
                    rows.append((layer.get_base_layer().weight,layer.lora_A[name].weight,layer.lora_B[name].weight,layer.scaling[name]))
                sequence.group.rows=rows;sequence.group.refresh()
            for module in linears:
                name=module.active_adapters[0]
                module._versioned_merged_weight.get(module.get_base_layer().weight,module.lora_A[name].weight,
                                                    module.lora_B[name].weight,module.scaling[name])
            for module in norms:
                w=module.weight;key=(id(w),w.data_ptr(),w._version,w.dtype,w.device)
                if getattr(module,'_fused_norm_revision',None)!=key:
                    module._unit_offset_weight=(1.+w.float()).detach();module._fused_norm_revision=key
        self.execution_cache.prepare(reference,revision,build)

    @contextmanager
    def _scope(self,prepared,reference):
        previous=self._kv_scope
        if previous is None or previous['reference']!=reference:self._prepare_execution(reference)
        with super()._scope(prepared,reference):yield

    def metadata(self):
        data=super().metadata()
        data.update(inference_layout='unpadded_full_verb_kv_grouped_v2',grouped_projections=self.group_counts,fused_mlp=self.fused_mlp_counts,
                    projection_arithmetic='versioned_grouped_BF16_projection_FP32_block_adapter_gradients')
        return data
