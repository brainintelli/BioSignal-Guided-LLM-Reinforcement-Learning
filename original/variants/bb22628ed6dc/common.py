import gc,os,time
from pathlib import Path
import torch
import base_smoke as b
from batch_boundary import bootstrap
from protocol import audit_gae
b.WORK=Path(__file__).resolve().parent;b.PROJECT=b.WORK/'project'

def identity(names):
    return dict(base=b.source_identity(),config=b.CONFIG,files={name:b.sha(b.WORK/name) for name in ['common.py','batch_boundary.py','protocol.py']+names})

def state(policy):
    return dict(actor=policy.digest(),critic=b.tensor_hash(policy.critic.named_parameters()),
        base=b.tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad),reference=b.tensor_hash(policy.reference_model.named_parameters()))

def resources(policy):
    policy.clear_prefix_cache();gc.collect();torch.cuda.synchronize()
    rss=int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rss=rss)

def environment():
    return b.GodotEnv(str(b.GODOT),str(b.PROJECT),map_mode=b.CONFIG['map_mode'],navigation_steps=b.CONFIG['navigation_steps'])

def sample(policy,env,tracker,obs,index,seed):
    pp=policy.prepare([obs]);s=policy.act(pp,with_reference=True);a=s['actions'][0];nxt=env.step(a)
    score=tracker.score(nxt['transition'],nxt['observation'])
    r={k:s[k+'s' if k in ('verb','target','value') else k][0] for k in ['verb','target','value','logp','verb_logp','reference_verb','reference_target']}
    r.update(prepared=pp[0],reward=score['reward'],duration=score['elapsed_ticks'],terminal=nxt['terminated'],boundary=nxt['terminated'])
    e=dict(index=index,seed=seed,observation=obs,prepared=b.plain(pp[0]),action=a,transition=nxt['transition'],next_observation=nxt['observation'],r1_score=score,
        reward_record=dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None))
    return r,e,nxt['observation']

def finish(policy,rows,obs):
    with torch.no_grad():boot,bp=bootstrap(rows,obs,lambda o:policy.prepare([o])[0],lambda p:policy.values([p])[0])
    b.finish_trajectory(rows,boot,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
    audit_gae(rows,boot,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
    return dict(value=boot,prepared=b.plain(bp) if bp is not None else None,terminal=rows[-1]['terminal'])

def record(rd,rows,events,boot):
    b.write(rd/'bootstrap.json',boot)
    with (rd/'rollout.jsonl').open('x') as f:
        for r,e in zip(rows,events):
            e['training']={k:v for k,v in r.items() if k!='prepared'};f.write(b.canonical(e)+'\n')

def statistics(policy,rows):
    return b.statistics(policy,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])

def checks(policy,initial,before,after,rows,input_hash):
    return dict(actor_changed=after['actor']!=before['actor'],critic_changed=after['critic']!=before['critic'],base_unchanged=after['base']==initial['base'],reference_unchanged=after['reference']==initial['reference'],
        inputs_unchanged=b.digest([b.plain(r['prepared']) for r in rows])==input_hash,
        finite_parameters=all(bool(torch.isfinite(p).all()) for m in [policy.model,policy.critic] for p in m.parameters()),
        finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for m in [policy.model,policy.critic] for p in m.parameters()),
        no_frozen_gradients=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad))

def optimizers(policy):
    return dict(actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=b.CONFIG['actor_lr'],eps=1e-5,weight_decay=0),
        critic=torch.optim.AdamW(policy.critic.parameters(),lr=b.CONFIG['critic_lr'],eps=1e-5,weight_decay=0))

def steps(opts):return {name:[int(s['step'].item()) for s in opt.state.values() if 'step' in s] for name,opt in opts.items()}

def manifest(out,name):b.write(out/name,{p.relative_to(out).as_posix():b.sha(p) for p in out.rglob('*') if p.is_file()})

def check_manifest(out,name):
    for file,h in b.read(out/name).items():assert b.sha(out/file)==h,file
