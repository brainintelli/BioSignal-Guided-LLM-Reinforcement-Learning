"""Synchronous8GPU PPO. Local rows stay on their sampling GPU."""
import json,math,time
import torch
import torch.distributed as dist

def global_assert(condition,message):
    failed=torch.tensor(int(not condition),dtype=torch.int32,device='cuda');dist.all_reduce(failed,op=dist.ReduceOp.MAX)
    assert not failed.item(),message

def gather_tensor(local):
    result=[torch.empty_like(local) for _ in range(dist.get_world_size())];dist.all_gather(result,local.contiguous());return torch.cat(result)

def global_normalize(local):
    values=gather_tensor(local);return (local-values.mean())/values.std(unbiased=False).clamp_min(1e-8)

def mean_scalar(value):
    t=torch.tensor(float(value),dtype=torch.float64,device='cuda');dist.all_reduce(t);return float(t/dist.get_world_size())

def max_scalar(value):
    t=torch.tensor(float(value),dtype=torch.float64,device='cuda');dist.all_reduce(t,op=dist.ReduceOp.MAX);return float(t)

def sync_gradients(parameters):
    ps=list(parameters);world=dist.get_world_size()
    global_assert(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in ps),'Nonfinite local gradient')
    used=torch.tensor([int(p.grad is not None) for p in ps],dtype=torch.int32,device='cuda');dist.all_reduce(used);used=used.tolist()
    for dtype in sorted({p.dtype for p in ps},key=str):
        group=[(i,p) for i,p in enumerate(ps) if p.dtype==dtype]
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for i,p in group])
        dist.all_reduce(flat);flat.div_(world);offset=0
        for i,p in group:
            part=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
            if used[i]==0:p.grad=None
            elif p.grad is None:p.grad=part.clone()
            else:p.grad.copy_(part)

def explained_variance(local_prediction,local_target):
    p=gather_tensor(local_prediction.detach());t=gather_tensor(local_target.detach());v=t.var(unbiased=False)
    return float(1-(t-p).var(unbiased=False)/v) if v>1e-10 else None

def update(policy,actor_optimizer,critic_optimizer,rows,c):
    from agent.m8_train import tensors
    world=dist.get_world_size();assert world==8 and len(rows)==16 and c.minibatch==32 and c.microbatch==4 and c.critic_minibatch==32
    local_batch=c.minibatch//world;policy.assert_reference_frozen();t=tensors(rows,policy.device_name);adv=global_normalize(t['advantage'])
    actor_parameters=[p for p in policy.model.parameters() if p.requires_grad];critic_parameters=list(policy.critic.parameters())
    with torch.no_grad():
        errors=[]
        for start in range(0,len(rows),c.microbatch):
            ix=list(range(start,min(start+c.microbatch,len(rows))))
            vd,td=policy.distributions([rows[i]['prepared'] for i in ix],t['verbs'][ix].tolist())
            errors.extend((vd.log_prob(t['verbs'][ix])+td.log_prob(t['targets'][ix])-t['logp'][ix]).tolist())
        zero=max_scalar(max(abs(x) for x in errors));global_assert(math.isfinite(zero) and zero<=c.zero_update_logp_tolerance,'Distributed old-policy probability drift')
        before=policy.values([r['prepared'] for r in rows]);ev_before=explained_variance(before,t['return'])
    metrics=[];stop=False;applied=0
    for epoch in range(c.epochs):
        order=torch.randperm(len(rows)).tolist()
        for start in range(0,len(rows),local_batch):
            ix=order[start:start+local_batch];assert len(ix)==c.microbatch
            actor_optimizer.zero_grad(set_to_none=True)
            vd,td=policy.distributions([rows[i]['prepared'] for i in ix],t['verbs'][ix].tolist())
            logp=vd.log_prob(t['verbs'][ix])+td.log_prob(t['targets'][ix]);logratio=logp-t['logp'][ix];ratio=logratio.exp()
            loss_policy=torch.maximum(-adv[ix]*ratio,-adv[ix]*ratio.clamp(1-c.clip,1+c.clip)).mean()
            importance=(vd.log_prob(t['verbs'][ix])-t['verb_logp'][ix]).exp()
            refv=t['reference_verb'][ix].clamp_min(1e-12);reft=t['reference_target'][ix].clamp_min(1e-12)
            vkl=(vd.probs*(vd.logits-refv.log())).sum(-1);tkl=(td.probs*(td.logits-reft.log())).sum(-1)
            reference_kl=(vkl+importance*tkl).mean();entropy=(vd.entropy()+importance*td.entropy()).mean()
            loss=loss_policy+c.reference_kl*reference_kl-c.entropy_coef*entropy
            global_assert(bool(torch.isfinite(loss)),'Nonfinite local actor objective');loss.backward();sync_gradients(actor_parameters)
            norm=float(torch.nn.utils.clip_grad_norm_(actor_parameters,c.actor_grad_clip));global_assert(math.isfinite(norm),'Nonfinite global actor gradient')
            local=dict(policy_loss=float(loss_policy.detach()),entropy=float(entropy.detach()),reference_kl=float(reference_kl.detach()),
                approx_kl=float(((ratio-1)-logratio).mean().detach()),clip_fraction=float(((ratio-1).abs()>c.clip).float().mean()),branch_importance=float(importance.mean().detach()))
            aggregate={k:mean_scalar(v) for k,v in local.items()};branch_max=max_scalar(float(importance.detach().max()))
            stop=aggregate['approx_kl']>c.target_kl
            if not stop:actor_optimizer.step();applied+=1
            metrics.append(dict(**aggregate,actor_grad_norm=norm,branch_importance_max=branch_max,applied=not stop))
            if dist.get_rank()==0:print(json.dumps(dict(event='joint_actor_minibatch',epoch=epoch,applied_steps=applied,kl=aggregate['approx_kl'],stopped=stop,time=time.time())),flush=True)
            if stop:break
        if stop:break
    critic_losses=[];critic_norms=[];critic_steps=0
    for epoch in range(c.critic_epochs):
        order=torch.randperm(len(rows)).tolist()
        for start in range(0,len(rows),local_batch):
            ix=order[start:start+local_batch];critic_optimizer.zero_grad(set_to_none=True)
            values=policy.values([rows[i]['prepared'] for i in ix]);loss=.5*(values-t['return'][ix]).square().mean()
            global_assert(bool(torch.isfinite(loss)),'Nonfinite critic loss');loss.backward();sync_gradients(critic_parameters)
            norm=float(torch.nn.utils.clip_grad_norm_(critic_parameters,c.critic_grad_clip));global_assert(math.isfinite(norm),'Nonfinite global critic gradient')
            critic_optimizer.step();critic_steps+=1;critic_losses.append(mean_scalar(float(loss.detach())));critic_norms.append(norm)
    policy.assert_reference_frozen()
    with torch.no_grad():ev_after=explained_variance(policy.values([r['prepared'] for r in rows]),t['return'])
    returns=gather_tensor(t['return']);advantages=gather_tensor(t['advantage'])
    return {**{k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0] if k!='applied'},
        'approx_kl_max':max(m['approx_kl'] for m in metrics),'branch_importance_max':max(m['branch_importance_max'] for m in metrics),
        'optimizer_steps':applied,'critic_optimizer_steps':critic_steps,'kl_early_stop':stop,'zero_update_logp_max':zero,
        'critic_loss':sum(critic_losses)/len(critic_losses),'critic_grad_norm':sum(critic_norms)/len(critic_norms),
        'explained_variance_before':ev_before,'explained_variance_after':ev_after,'advantage_std':float(advantages.std(unbiased=False)),
        'return_mean':float(returns.mean()),'return_std':float(returns.std(unbiased=False))}
