"""One-timeline synchronous PPO and stride-preserving activation offload.

The objective follows agent.m8_train.update. Gradient SUM is intentional:
each rank contributes sum(local losses)/global minibatch size. There are no
rank-local rollouts and no duplication of the biological sample count.
"""
from contextlib import contextmanager
import math


def shard_minibatch(indices, rank, world):
    if not 0 <= rank < world:raise ValueError('invalid rank/world')
    return indices[rank::world]


class Collective:
    def __init__(self, device, rank=0, world=1):
        self.device, self.rank, self.world = device, rank, world

    def require(self, condition, message):
        import torch
        failed=torch.tensor(int(not condition),dtype=torch.int32,device=self.device)
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if failed.item():raise RuntimeError(message)

    def sum(self, value):
        import torch
        t=torch.tensor(value,dtype=torch.float64,device=self.device)
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(t)
        return t.tolist()

    def maximum(self, value):
        import torch
        t=torch.tensor(float(value),dtype=torch.float64,device=self.device)
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(t,op=dist.ReduceOp.MAX)
        return float(t)

    def gradients(self, parameters):
        import torch
        parameters=list(parameters)
        self.require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters),
                     'nonfinite local gradients')
        for p in parameters:
            # Unused branches / empty ranks still take part in every collective.
            if p.grad is None:p.grad=torch.zeros_like(p)
            if self.world > 1:
                import torch.distributed as dist
                dist.all_reduce(p.grad)

    def identical(self, value, label):
        if self.world == 1:return
        import torch.distributed as dist
        values=[None]*self.world
        dist.all_gather_object(values,value)
        self.require(all(v == values[0] for v in values),label+' differs across replicas')


@contextmanager
def saved_on_cpu_preserve_layout():
    """Frozen from the verified M9 offload helper; retains padded SDPA strides."""
    import torch
    def pack(tensor):
        if tensor.device.type != 'cuda':return (None,tensor.detach())
        if tensor.layout != torch.strided or any(s < 0 for s in tensor.stride()):
            raise ValueError('nonnegative strided tensors required')
        size,stride=tuple(tensor.size()),tuple(tensor.stride())
        span=0 if tensor.numel()==0 else 1+sum((n-1)*s for n,s in zip(size,stride))
        flat=tensor.detach().as_strided((span,),(1,),tensor.storage_offset())
        host=torch.empty(span,dtype=tensor.dtype,device='cpu',pin_memory=True)
        host.copy_(flat)
        return tensor.device,host,size,stride
    def unpack(packed):
        device,host,*metadata=packed
        if device is None:return host
        size,stride=metadata
        return host.to(device,non_blocking=True).as_strided(size,stride,0)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):yield


def ppo_update(policy, optimizers, rows, c, collective, update_id):
    import torch
    names=('advantage','return','value','logp','verb_logp','reference_verb','reference_target')
    t={k:torch.tensor([r[k] for r in rows],device=policy.device_name,dtype=torch.float32) for k in names}
    t.update(verbs=torch.tensor([r['verb'] for r in rows],device=policy.device_name),
             targets=torch.tensor([r['target'] for r in rows],device=policy.device_name))
    adv=t['advantage'];adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    actor=list(p for p in policy.model.parameters() if p.requires_grad)
    critic=list(policy.critic.parameters())
    # A dedicated identical generator cannot be perturbed by rank-zero sampling.
    generator=torch.Generator(device='cpu').manual_seed(c.rng_seed+update_id)
    actor_metrics=[];actor_steps=critic_steps=0;stop=False
    fields=('policy_loss','entropy','reference_kl','approx_kl','clip_fraction','branch_importance')
    def batches(size):
        order=torch.randperm(len(rows),generator=generator).tolist()
        for start in range(0,len(rows),size):yield order[start:start+size]
    for _ in range(c.epochs):
        for global_ix in batches(c.minibatch):
            local_ix=shard_minibatch(global_ix,collective.rank,collective.world)
            optimizers['actor'].zero_grad(set_to_none=True)
            aggregate=[0.]*len(fields);branch_max=0.
            finite=True
            for start in range(0,len(local_ix),c.microbatch):
                ix=local_ix[start:start+c.microbatch]
                vd,td=policy.distributions([rows[i]['prepared'] for i in ix],t['verbs'][ix].tolist())
                logp=vd.log_prob(t['verbs'][ix])+td.log_prob(t['targets'][ix])
                logratio=logp-t['logp'][ix];ratio=logratio.exp()
                policy_loss=torch.maximum(-adv[ix]*ratio,-adv[ix]*ratio.clamp(1-c.clip,1+c.clip)).mean()
                importance=(vd.log_prob(t['verbs'][ix])-t['verb_logp'][ix]).exp()
                refv=t['reference_verb'][ix].clamp_min(1e-12);reft=t['reference_target'][ix].clamp_min(1e-12)
                reference_kl=((vd.probs*(vd.logits-refv.log())).sum(-1)+
                    importance*(td.probs*(td.logits-reft.log())).sum(-1)).mean()
                entropy=(vd.entropy()+importance*td.entropy()).mean()
                loss=policy_loss+c.reference_kl*reference_kl-c.entropy_coef*entropy
                finite=finite and bool(torch.isfinite(loss))
                weight=len(ix)/len(global_ix)
                if finite:(loss*weight).backward()
                vals=(policy_loss,entropy,reference_kl,((ratio-1)-logratio).mean(),
                      ((ratio-1).abs()>c.clip).float().mean(),importance.mean())
                for j,v in enumerate(vals):aggregate[j]+=float(v.detach())*weight
                branch_max=max(branch_max,float(importance.detach().max()))
            collective.require(finite,'nonfinite actor objective')
            collective.gradients(actor)
            norm=float(torch.nn.utils.clip_grad_norm_(actor,c.actor_grad_clip))
            collective.require(math.isfinite(norm),'nonfinite actor gradient norm')
            metric=dict(zip(fields,collective.sum(aggregate)))
            metric.update(actor_grad_norm=norm,branch_importance_max=collective.maximum(branch_max))
            stop=metric['approx_kl']>c.target_kl
            if not stop:optimizers['actor'].step();actor_steps+=1
            actor_metrics.append(metric)
            if stop:break
        if stop:break
    critic_losses=[];critic_norms=[]
    for _ in range(c.critic_epochs):
        for global_ix in batches(c.critic_minibatch):
            local_ix=shard_minibatch(global_ix,collective.rank,collective.world)
            optimizers['critic'].zero_grad(set_to_none=True)
            loss_value=0.;finite=True
            if local_ix:
                values=policy.values([rows[i]['prepared'] for i in local_ix])
                loss=.5*(values-t['return'][local_ix]).square().sum()/len(global_ix)
                finite=bool(torch.isfinite(loss))
                if finite:loss.backward()
                loss_value=float(loss.detach())
            collective.require(finite,'nonfinite critic loss');collective.gradients(critic)
            norm=float(torch.nn.utils.clip_grad_norm_(critic,c.critic_grad_clip))
            collective.require(math.isfinite(norm),'nonfinite critic gradient norm')
            optimizers['critic'].step();critic_steps+=1
            critic_losses.append(collective.sum(loss_value));critic_norms.append(norm)
    policy.assert_reference_frozen()
    return {**{k:sum(m[k] for m in actor_metrics)/len(actor_metrics) for k in actor_metrics[0]},
            'approx_kl_max':max(m['approx_kl'] for m in actor_metrics),
            'optimizer_steps':actor_steps,'critic_optimizer_steps':critic_steps,'kl_early_stop':stop,
            'critic_loss':sum(critic_losses)/len(critic_losses),
            'critic_grad_norm':sum(critic_norms)/len(critic_norms),
            'unique_decisions':len(rows),'world_size':collective.world,
            'advantage_std':float(t['advantage'].std(unbiased=False)),
            'return_mean':float(t['return'].mean()),'return_std':float(t['return'].std(unbiased=False))}


def _init_group(config, rank, rendezvous):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    torch.cuda.set_device(config['devices'][rank])
    dist.init_process_group('nccl',init_method=rendezvous,rank=rank,
                            world_size=len(config['devices']),timeout=timedelta(seconds=240))


def receive_command(control):
    # Do not start the NCCL timeout while rank zero is collecting observations.
    # A dead parent closes this OS pipe; no device/acquisition channel is used.
    if control.recv()!='broadcast':raise ValueError('invalid parent control cue')
    import torch.distributed as dist
    message=[None];dist.broadcast_object_list(message,src=0)
    return message[0]


def _worker(rank,config,rendezvous,output_dir,control):
    """A worker has a model and optimizer, but no Godot/source/client instance."""
    import json
    import traceback
    from pathlib import Path
    import torch.distributed as dist
    from .m9_runtime import _ModelCore, _build_policy, _difference
    try:
        _init_group(config,rank,rendezvous)
        core=_ModelCore(_build_policy(config,config['devices'][rank]),config,rank,len(config['devices']))
        while True:
            message=receive_command(control)
            if message['op']=='close':break
            if message['op']=='update':
                core.update(message['rows'],message['bootstrap'],message['update_id'])
            elif message['op']=='checkpoint':
                before=core.state();optimizer=core.optimizer_identity();stats=core.statistics(core.last_rows)
                dist.barrier()  # rank zero publishes a complete atomic checkpoint
                core.load_checkpoint(message['path'],message['registration_sha'])
                error=_difference(stats,core.statistics(core.last_rows))
                core.collective.require(before==core.state() and optimizer==core.optimizer_identity() and error<=1e-6,
                                        'worker checkpoint roundtrip')
                core.collective.identical(dict(model=core.state(),optimizer=core.optimizer_identity()),'checkpoint reload')
            else:raise ValueError('unknown replica command')
    except BaseException as error:
        path=Path(output_dir)/f'runtime_worker_{rank}_failure.json'
        path.write_text(json.dumps(dict(rank=rank,error=str(error),traceback=traceback.format_exc()),indent=2),encoding='utf-8')
        raise
    finally:
        control.close()
        if dist.is_initialized():dist.destroy_process_group()


class ReplicaGroup:
    """Parent-owned local workers. This is never created by offline-verify."""
    def __init__(self,config,output_dir):
        import multiprocessing
        import uuid
        from pathlib import Path
        import torch
        if len(config['devices'])!=8:raise ValueError('eight replica devices required')
        if any(d>=torch.cuda.device_count() for d in config['devices']):raise ValueError('configured GPU absent')
        self.closed=False;self.processes=[];self.controls=[]
        rendezvous=(Path(output_dir)/('runtime_rendezvous_'+uuid.uuid4().hex)).as_uri()
        context=multiprocessing.get_context('spawn')
        try:
            for rank in range(1,8):
                child,parent=context.Pipe(duplex=False)
                process=context.Process(target=_worker,args=(rank,config,rendezvous,str(output_dir),child),
                                        name=f'mea-model-replica-{rank}')
                process.start();child.close()
                self.processes.append(process);self.controls.append(parent)
            _init_group(config,0,rendezvous)
        except BaseException:
            self.abort();raise

    def send(self,message):
        import torch.distributed as dist
        if self.closed or any(not p.is_alive() for p in self.processes):
            raise RuntimeError('model replica exited')
        for control in self.controls:control.send('broadcast')
        dist.broadcast_object_list([message],src=0)

    def checkpoint_barrier(self):
        import torch.distributed as dist
        dist.barrier()

    def close(self):
        if self.closed:return
        try:
            self.send(dict(op='close'))
            for process in self.processes:
                process.join(timeout=10)
                if process.is_alive() or process.exitcode!=0:raise RuntimeError('replica cleanup failed')
        finally:self.abort()

    def abort(self):
        import torch.distributed as dist
        if self.closed:return
        self.closed=True
        for control in self.controls:control.close()
        for process in self.processes:
            if process.is_alive():process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():process.kill();process.join(timeout=5)
        if dist.is_initialized():dist.destroy_process_group()
