"""Explicit GPU smoke: parent fidelity, real 8-decision PPO, fixed-shape ratios and reload.

Only reads actual open3 Godot development observations. Never uses held-out observations or changes
the released adapter. All updates belong to a new disposable smoke directory.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import subprocess
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def layout_indices(count, anchor):
    if count < 4 or not 0 <= anchor < count:
        raise ValueError('At least four actual observations and a valid anchor are required')
    cases=[]
    for variant in range(2):
        for slot in range(4):
            indices=[(anchor+1+variant*3+j)%count for j in range(4)]
            indices[slot]=anchor
            cases.append((variant,slot,indices))
    return cases


def check_layouts(policy, observations):
    """Exact dynamic-slot/filler invariance, all legal conditional branches."""
    import hashlib
    import torch
    prepared=policy.prepare(observations)
    cases=[]
    with torch.no_grad():
        for anchor_index in range(len(prepared)):
            anchor=prepared[anchor_index]
            single_v=policy.verb_distribution([anchor]).probs[0]
            legal=[i for i,yes in enumerate(anchor['verb_mask']) if yes]
            singles={v:policy.target_distribution([anchor],[v]).probs[0] for v in legal}
            for variant,slot,indices in layout_indices(len(prepared),anchor_index):
                    batch=[prepared[j] for j in indices]
                    bv=policy.verb_distribution(batch).probs
                    if not torch.equal(bv[slot],single_v):raise RuntimeError('Refill verb probability mismatch')
                    for verb in legal:
                        choices=[next(i for i,yes in enumerate(r['verb_mask']) if yes) for r in batch]
                        choices[slot]=verb
                        bt=policy.target_distribution(batch,choices).probs[slot]
                        if not torch.equal(bt,singles[verb]):raise RuntimeError('Refill conditional target probability mismatch')
                    ga=torch.Generator(device=policy.device_name).manual_seed(800000+anchor_index)
                    gb=torch.Generator(device=policy.device_name).manual_seed(800000+anchor_index)
                    va=int(torch.multinomial(single_v,1,generator=ga))
                    vb=int(torch.multinomial(bv[slot],1,generator=gb))
                    choices=[next(i for i,yes in enumerate(r['verb_mask']) if yes) for r in batch];choices[slot]=vb
                    bt=policy.target_distribution(batch,choices).probs[slot]
                    ta=int(torch.multinomial(singles[va],1,generator=ga));tb=int(torch.multinomial(bt,1,generator=gb))
                    if (va,ta)!=(vb,tb) or not torch.equal(ga.get_state(),gb.get_state()):raise RuntimeError('Refill action/RNG mismatch')
                    cases.append(dict(anchor=anchor_index,slot=slot,filler_variant=variant,legal_verbs=legal,
                        verb=va,target=ta,rng_sha256=hashlib.sha256(ga.get_state().cpu().numpy().tobytes()).hexdigest()))
    first_lengths=[len(r['input_ids']) for r in prepared]
    target_lengths=[len(ids) for r in prepared for ids in r['target_input_ids'].values()]
    return dict(passed=True,cases=cases,observations=len(prepared),
        first_input_tokens_max=max(first_lengths),target_input_tokens_max=max(target_lengths,default=0),
        all_conditional_branches_checked=True,exact_rng_bytes=True,
        scope='Finite actual training observation pool; not an all-state guarantee')


def main():
    import torch
    from agent.m8x_policy import Policy
    from agent.m8x_env import GodotEnv
    from agent.m8x_contract import SurvivalReward
    from agent.m8_train import update
    from agent.rl_contract import gae
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-gpu-probe',action='store_true',help='Explicit post-freeze smoke authorization')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.allow_gpu_probe:raise SystemExit("GPU probe disabled without explicit post-freeze authorization")
    config=json.loads(Path(args.config).read_text());c=SimpleNamespace(**config)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('Exclusive GPU required; processes untouched: '+active)
    started=time.monotonic()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    parent=Path(c.parent_checkpoint)
    if c.profile != 'open3' or c.train_seed_start != 88080 or c.gamma_tick != .9995:
        raise ValueError('Probe requires the explicit balanced training configuration')
    envs=[GodotEnv(c.godot,c.project) for _ in range(4)]
    try:
        states=[e.reset(c.train_seed_start+i,c.profile) for i,e in enumerate(envs)]
        observations=[s['observation'] for s in states]
    finally:
        for e in envs:e.close()
    (output/'initial_observations.json').write_text(json.dumps(observations,indent=2))
    policy=Policy(c.model,parent_checkpoint=parent,seed=c.seed,max_tokens=c.max_tokens)
    initial=policy.digest();policy.assert_reference_frozen()
    if len(policy.target_ids)!=64 or len(set(policy.target_ids))!=64:
        raise RuntimeError('Balanced interface requires 64 distinct single target tokens')
    layout_before=check_layouts(policy,observations)
    prepared=policy.prepare(observations)
    with torch.no_grad():
        verbs=policy.verb_distribution(prepared).probs.argmax(-1).tolist()
        vd,td=policy.distributions(prepared,verbs)
        rv,rt=policy.distributions(prepared,verbs,reference=True)
        verb_diff=float((vd.probs-rv.probs).abs().max())
        target_diff=float((td.probs-rt.probs).abs().max())
        if verb_diff!=0 or target_diff!=0:raise RuntimeError('Actor/reference initially differ on semantic interface')
        for row in prepared:
            for legal_verb,allowed in enumerate(row['verb_mask']):
                if not allowed:continue
                av,at=policy.distributions([row],[legal_verb])
                pv,pt=policy.distributions([row],[legal_verb],reference=True)
                if not torch.equal(av.probs,pv.probs) or not torch.equal(at.probs,pt.probs):
                    raise RuntimeError('Parent/reference differs on a legal branch')
        del av,at,pv,pt
        batch_diff=0.;rng_equal=True
        for i,row in enumerate(prepared):
            sv,st=policy.distributions([row],[verbs[i]])
            batch_diff=max(batch_diff,float((sv.probs[0]-vd.probs[i]).abs().max()),float((st.probs[0]-td.probs[i]).abs().max()))
            ga=torch.Generator(device=policy.device_name).manual_seed(170000+i)
            gb=torch.Generator(device=policy.device_name).manual_seed(170000+i)
            va=int(torch.multinomial(vd.probs[i],1,generator=ga));vb=int(torch.multinomial(sv.probs[0],1,generator=gb))
            # Recompute selected conditional target for the sampled verb in both layouts.
            batch_verbs=list(verbs);batch_verbs[i]=va
            bat=policy.target_distribution(prepared,batch_verbs).probs[i]
            one=policy.target_distribution([row],[vb]).probs[0]
            ta=int(torch.multinomial(bat,1,generator=ga));tb=int(torch.multinomial(one,1,generator=gb))
            batch_diff=max(batch_diff,float((bat-one).abs().max()))
            rng_equal &= va==vb and ta==tb and torch.equal(ga.get_state(),gb.get_state())
        if batch_diff!=0 or not rng_equal:raise RuntimeError('Batched/single live probabilities or RNG differ')
    del vd,td,rv,rt,sv,st
    envs=[GodotEnv(c.godot,c.project) for _ in range(4)]
    rows=[]
    try:
        states=[e.reset(c.train_seed_start+i,c.profile) for i,e in enumerate(envs)]
        trackers=[SurvivalReward(s['observation']) for s in states]
        trajectories=[[] for _ in envs]
        for _ in range(2):
            prepared=policy.prepare([s['observation'] for s in states])
            sampled=policy.act(prepared,with_reference=True)
            for i,e in enumerate(envs):
                nxt=e.step(sampled['actions'][i]);reward=trackers[i].score(nxt['transition'],nxt['observation'])
                trajectories[i].append(dict(prepared=prepared[i],verb=sampled['verbs'][i],target=sampled['targets'][i],
                    value=sampled['values'][i],logp=sampled['logp'][i],verb_logp=sampled['verb_logp'][i],
                    reference_verb=sampled['reference_verb'][i],reference_target=sampled['reference_target'][i],
                    reward=reward['reward'],duration=reward['elapsed_ticks'],terminal=nxt['terminated'],boundary=nxt['terminated'],
                    next_obs=nxt['observation']))
                with (output/'smoke_transitions.jsonl').open('a') as log:
                    log.write(json.dumps(dict(env=i,observation=states[i]['observation'],action=sampled['actions'][i],
                                              transition=nxt['transition'],next_observation=nxt['observation'],reward=reward))+'\n')
                states[i]=nxt
        for trajectory in trajectories:
            for row in trajectory:
                with torch.no_grad():row['next_value']=0. if row['terminal'] else float(policy.values(policy.prepare([row['next_obs']]))[0])
            trajectory[-1]['boundary']=True
            advantages,returns=gae([r['reward'] for r in trajectory],[r['value'] for r in trajectory],
                [r['next_value'] for r in trajectory],[r['duration'] for r in trajectory],
                [r['terminal'] for r in trajectory],[r['boundary'] for r in trajectory],gamma_tick=c.gamma_tick,lambda_tick=c.gae_lambda)
            for row,adv,ret in zip(trajectory,advantages,returns):row.update(advantage=adv,**{'return':ret});rows.append(row)
    finally:
        for e in envs:e.close()
    prepared=[r['prepared'] for r in rows[:4]];verbs=[r['verb'] for r in rows[:4]]
    targets=torch.tensor([r['target'] for r in rows[:4]],device=policy.device_name)
    v=torch.tensor(verbs,device=policy.device_name)
    with torch.no_grad():
        vd,td=policy.distributions(prepared,verbs);old_logp=vd.log_prob(v)+td.log_prob(targets)
    vd,td=policy.distributions(prepared,verbs)
    grad_drift=float(((vd.log_prob(v)+td.log_prob(targets)).detach()-old_logp).abs().max())
    del vd,td,old_logp
    if grad_drift!=0:raise RuntimeError('Grad/no-grad logp differs')
    c.epochs=1;c.minibatch=8;c.critic_epochs=1;c.critic_minibatch=8
    actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=c.actor_lr,eps=1.e-5,weight_decay=0.)
    critic=torch.optim.AdamW(policy.critic.parameters(),lr=c.critic_lr,eps=1.e-5,weight_decay=0.)
    torch.cuda.synchronize();update_started=time.monotonic()
    metrics=update(policy,actor,critic,rows,c)
    torch.cuda.synchronize();update_seconds=time.monotonic()-update_started
    if metrics['zero_update_logp_max']!=0 or metrics['optimizer_steps']!=1:
        raise RuntimeError('Smoke must have exact ratios and one actual actor update')
    policy.assert_reference_frozen()
    trained=policy.digest()
    if trained==initial:raise RuntimeError('Smoke optimizer did not change actor')
    layout_after=check_layouts(policy,observations+[r['next_obs'] for r in rows])
    policy.save(output/'checkpoint')
    with torch.no_grad():
        vd,td=policy.distributions(prepared,verbs);saved_v=vd.probs.cpu();saved_t=td.probs.cpu()
    peak=torch.cuda.max_memory_allocated()
    if peak>=48*1024**3:raise RuntimeError('Smoke exceeds the registered physical memory ceiling')
    del policy,actor,critic,vd,td;gc.collect();torch.cuda.empty_cache()
    restored=Policy(c.model,checkpoint=output/'checkpoint',parent_checkpoint=parent,seed=c.seed,max_tokens=c.max_tokens)
    with torch.no_grad():
        vd,td=restored.distributions(prepared,verbs)
        reload_diff=max(float((vd.probs.cpu()-saved_v).abs().max()),float((td.probs.cpu()-saved_t).abs().max()))
    if reload_diff!=0 or restored.digest()!=trained:raise RuntimeError('Checkpoint reload changed policy')
    restored.assert_reference_frozen()
    result=dict(passed=True,scope='discarded_open3_development_probe',total_seconds=time.monotonic()-started,update_seconds=update_seconds,release_authorized=False,survival_evaluated=False,formal_resume_permitted=False,reference_actor_sha256=restored.reference_digest(),parent_actor_sha256=initial,trained_smoke_actor_sha256=trained,parent_verb_max_difference=verb_diff,
        parent_target_max_difference=target_diff,batched_single_max_difference=batch_diff,batched_single_rng_equal=rng_equal,grad_logp_max_difference=grad_drift,
        reload_max_difference=reload_diff,metrics=metrics,peak_cuda_bytes=peak,actual_environment_decisions=8,
        layout_before=layout_before,layout_after=layout_after,heldout_observations_used=False,production_checkpoint_modified=False,policy=restored.metadata())
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
