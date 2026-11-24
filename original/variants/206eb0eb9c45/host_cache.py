"""Release unused pinned-host cache at an already completed update boundary.

Pinned to this task's authenticated PyTorch2.6 CUDA runtime. This internal API
must be revalidated before changing runtimes; it never frees live tensors.
"""
import gc
import torch

def release_host_cache():
    assert str(torch.__version__)=='2.6.0+cu124',torch.__version__
    assert hasattr(torch._C,'_host_emptyCache')
    torch.cuda.synchronize();gc.collect();torch._C._host_emptyCache()
