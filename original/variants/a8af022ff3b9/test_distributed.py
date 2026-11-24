"""Actual eight-rank math test: gradient scale, AdamW, missing grads and KL."""
from distributed_update import global_normalize,sync_gradients,mean_scalar,global_assert
import json,os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180),device_id=torch.device('cuda',rank))
    assert dist.get_world_size()==8
    local=torch.arange(rank*4,rank*4+4,dtype=torch.float64,device='cuda')
    all_values=torch.arange(32,dtype=torch.float64,device='cuda')
    expected=(all_values-all_values.mean())/all_values.std(unbiased=False)
    actual=global_normalize(local)
    assert torch.allclose(actual,expected[rank*4:rank*4+4],atol=1e-12,rtol=0)
    p=torch.nn.Parameter(torch.tensor([.25,-.5],dtype=torch.float64,device='cuda'))
    ref=torch.nn.Parameter(p.detach().clone());x=torch.stack([local,local.square()/31],1);y=local/7
    loss=(x@p-y).square().mean();loss.backward();sync_gradients([p])
    xx=torch.stack([all_values,all_values.square()/31],1);yy=all_values/7
    (xx@ref-yy).square().mean().backward()
    err=float((p.grad-ref.grad).abs().max());assert err<1e-10,err
    a=torch.optim.AdamW([p],lr=.001,eps=1e-5,weight_decay=0);b=torch.optim.AdamW([ref],lr=.001,eps=1e-5,weight_decay=0)
    torch.nn.utils.clip_grad_norm_([p],1.);torch.nn.utils.clip_grad_norm_([ref],1.);a.step();b.step()
    update_error=float((p-ref).abs().max());assert update_error<1e-12
    partial=torch.nn.Parameter(torch.ones(2,device='cuda'));unused=torch.nn.Parameter(torch.ones(1,device='cuda'))
    if rank==0:partial.grad=torch.tensor([8.,16.],device='cuda')
    sync_gradients([partial,unused]);assert torch.equal(partial.grad,torch.tensor([1.,2.],device='cuda')) and unused.grad is None
    kl=mean_scalar(.16 if rank==7 else 0.);assert abs(kl-.02)<1e-12 and kl>.015
    # Every rank must reject a failing condition observed only on rank7.
    rejected=False
    try:global_assert(rank!=7,'injected rank-local failure')
    except AssertionError:rejected=True
    assert rejected
    results=[None]*8;dist.all_gather_object(results,dict(rank=rank,pid=os.getpid(),device=rank,gradient_error=err,update_error=update_error,global_kl=kl,passed=True))
    if rank==0:
        with (Path(os.environ['TASK_ROOT'])/'distributed_math.json').open('x') as f:json.dump(dict(passed=True,ranks=results),f,indent=2)
        print(json.dumps(dict(event='distributed_math_passed',ranks=8,max_gradient_error=max(r['gradient_error'] for r in results))))
    dist.destroy_process_group()
if __name__=='__main__':main()
