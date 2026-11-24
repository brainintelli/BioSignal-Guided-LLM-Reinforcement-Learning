"""GPU context/reuse checks. Synthetic stress is not a survival evaluation."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.m8s3_training import sha, write
from tools.m9_campaign import task_spec, verify_registration


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--previous', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--reload', action='store_true')
    a=ap.parse_args()
    previous=json.loads(Path(a.previous).read_text());verify_registration(previous)
    from tools.benchmark_m8s3 import setup, selected_logps
    setup()
    import torch
    from agent.m9_policy import Policy, load, save
    project=Path(__file__).resolve().parents[1]
    spec=task_spec(previous,'explore','state_compact',1)
    spec.update(project=str(project), max_tokens=4096)
    c=SimpleNamespace(**spec)
    out=Path(a.output)
    policy=Policy(c)
    if a.reload:
        load(policy,out/'checkpoint')
        probe=json.loads((out/'long_probe.json').read_text())
        prepared=policy.prepare(probe['observations'])
        actual=selected_logps(policy,prepared,probe['verbs'],probe['targets'])
        error=max(abs(x-y) for x,y in zip(actual,probe['logp']))
        if error>1e-6:raise ValueError('Independent long-context reload drift')
        write(out/'independent_reload.json',dict(passed=True,logp_error=error,actor_sha256=policy.digest()))
        print(json.dumps(dict(independent_reload_error=error)),flush=True)
        return
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for rep in previous['config']['phases']['explore']['replicates']:
        for arm in previous['config']['arms']:
            task=task_spec(previous,'explore',arm,rep)
            cp=Path(task['output'])/'checkpoint_0008'
            if not cp.exists():continue
            task.update(project=str(project),max_tokens=4096)
            policy.c=SimpleNamespace(**task);load(policy,cp)
            probe=json.loads((cp/'reload_probe.json').read_text())
            rows=probe['rows'];prepared=policy.prepare([r['observation'] for r in rows])
            actual=selected_logps(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows])
            error=max(abs(x-y) for x,y in zip(actual,probe['logp']))
            if error>1e-6:raise ValueError('Prior checkpoint probability changed: '+str(cp))
            results.append(dict(arm=arm,replicate=rep,checkpoint=str(cp),rows=len(rows),error=error,actor_sha256=policy.digest()))
            print(json.dumps(results[-1]),flush=True)
    policy.c=c;load(policy,Path(spec['output'])/'checkpoint_0008')
    fixture=json.loads((project/'tests/fixtures/m9_token_overflow.json').read_text())
    obs=fixture['observation']
    policy.max_tokens=1536
    try:policy.prepare([obs])
    except ValueError as exc:
        if '1541 exceeds 1536' not in str(exc):raise
        original_error=str(exc)
    else:raise AssertionError('Original failure did not reproduce')
    policy.max_tokens=4096
    observations=[copy.deepcopy(obs) for _ in range(4)]
    for i,item in enumerate(observations):item['player']['thirst']-=i*.08
    prepared=policy.prepare(observations)
    verbs=[next(i for i,b in enumerate(r['branches']) if len(b)>1) for r in prepared]
    targets=[0]*4
    before=selected_logps(policy,prepared,verbs,targets)
    regrouped=selected_logps(policy,prepared,verbs,targets,size=1,order=[3,1,0,2],grad=True)
    error=max(abs(x-y) for x,y in zip(before,regrouped))
    if error>1e-6:raise ValueError('Long context peer/grad layout drift')
    stress=[]
    for width in (None,2048,3072,4032):
        rows=copy.deepcopy(prepared)
        if width:
            for i,row in enumerate(rows):
                ids=row['input_ids'];row['input_ids']=ids+[policy.tokenizer.eos_token_id]*(width-len(ids))
                row['input_ids'][-1]=ids[-1-i]
        policy.model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        vd,td=policy.distributions(rows,verbs)
        v=torch.tensor(verbs,device='cuda');t=torch.tensor(targets,device='cuda')
        loss=-(vd.log_prob(v)+td.log_prob(t)).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite context stress loss')
        loss.backward()
        grads=[p.grad for p in policy.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Invalid context gradients')
        torch.cuda.synchronize()
        stress.append(dict(synthetic=bool(width),tokens=[len(r['input_ids']) for r in rows],
            loss=float(loss.detach()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite_gradients=True))
        print(json.dumps(stress[-1]),flush=True)
        del vd,td,loss,grads,rows
    policy.model.zero_grad(set_to_none=True)
    actor_opt=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=c.actor_lr)
    critic_opt=torch.optim.AdamW(policy.critic.parameters(),lr=c.critic_lr)
    save(policy,out/'checkpoint',actor_opt,critic_opt,8,previous['registration_sha256'])
    write(out/'long_probe.json',dict(observations=observations,verbs=verbs,targets=targets,logp=before))
    write(out/'summary.json',dict(passed=True,original_error=original_error,checkpoint_probes=results,
        extended_layout=policy.bucket_boundaries,long_probability_error=error,stress=stress,
        optimization_steps=0,source_sha256=sha(__file__),previous_registration_sha256=previous['registration_sha256']))


if __name__=='__main__':main()
