"""CPU saved-tensor storage preserving strides, including padded SDPA biases.

Copy the addressed storage span rather than compacting a view. This preserves
padding and broadcast strides when restoring for backward. No dtype conversion.
"""
import torch

@torch.no_grad()
def pack(tensor):
    if tensor.device.type!='cuda':return (None,tensor.detach())
    if tensor.layout!=torch.strided or any(s<0 for s in tensor.stride()):
        raise ValueError('Only nonnegative strided CUDA tensors are supported')
    size=tuple(tensor.size());stride=tuple(tensor.stride())
    span=0 if tensor.numel()==0 else 1+sum((n-1)*s for n,s in zip(size,stride))
    flat=tensor.detach().as_strided((span,),(1,),tensor.storage_offset())
    host=torch.empty(span,dtype=tensor.dtype,device='cpu',pin_memory=True)
    host.copy_(flat)  # Complete D2H before the saved tensor may be unpacked.
    return (tensor.device,host,size,stride)

@torch.no_grad()
def unpack(packed):
    device,host,*metadata=packed
    if device is None:return host
    size,stride=metadata
    storage=host.to(device,non_blocking=True)
    return storage.as_strided(size,stride,0)

def saved_on_cpu_preserve_layout():
    return torch.autograd.graph.saved_tensors_hooks(pack,unpack)
