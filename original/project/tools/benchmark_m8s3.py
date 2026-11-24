"""Bounded equal-update open4 experiment. No model/optimizer checkpoint writes."""
import argparse,gc,hashlib,json,math,random,statistics,subprocess,sys,time,warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.m8s3_diag_contract import validate_config,verify_registration_digest,speed_gate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def emit(**value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
def verify(reg):
    verify_registration_digest(reg);validate_config(reg['config'])
    for path,digest in reg['source_hashes'].items():
        if sha(Path(reg['config']['project'])/path)!=digest:raise ValueError('Source changed: '+path)
    for item in reg['input_files'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Input changed: '+item['path'])
def setup():
    warnings.filterwarnings('ignore',message='Input tensor shape suggests potential format mismatch:.*',module=r'fla\.ops\.gated_delta_rule\.chunk')
    import torch
    torch.set_num_threads(8)
def make(c,mode):
    from agent.m8s3_policy import BatchedSemanticPolicy
    from agent.m8s3_padded_policy import PaddedSemanticPolicy,NativePaddedSemanticPolicy
    baseline=NativePaddedSemanticPolicy if c.baseline_projection_layout=='native' else PaddedSemanticPolicy
    cls=baseline if mode=='baseline' else BatchedSemanticPolicy
    return cls(c.model,checkpoint=c.initial_checkpoint,parent_checkpoint=c.parent_checkpoint,seed=c.rng_seed,
               reset_critic=True,with_reference=True,max_tokens=c.max_tokens,prefix_layout=c.prefix_layout)
def critic_digest(p):
    h=hashlib.sha256()
    for name,value in p.critic.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def difference(a,b):
    diffs=[];kls=[];changed=0
    for x,y in zip(a,b):
        assert len(x)==len(y)
        diffs.append(max(abs(u-v) for u,v in zip(x,y)))
        kls.append(sum(u*math.log(max(u,1e-30)/max(v,1e-30)) for u,v in zip(x,y)))
        changed+=max(range(len(x)),key=x.__getitem__)!=max(range(len(y)),key=y.__getitem__)
    return dict(max_abs=max(diffs),mean_kl=statistics.mean(kls),max_kl=max(kls),argmax_changes=changed)
def selected_logps(p,prepared,verbs,targets,size=4,order=None,grad=False):
    import torch
    result=[None]*len(prepared);order=list(range(len(prepared))) if order is None else order
    with torch.set_grad_enabled(grad):
        for start in range(0,len(order),size):
            ix=order[start:start+size];v=torch.tensor([verbs[i] for i in ix],device='cuda');t=torch.tensor([targets[i] for i in ix],device='cuda')
            vd,td=p.distributions([prepared[i] for i in ix],v.tolist())
            for i,value in zip(ix,(vd.log_prob(v)+td.log_prob(t)).detach().tolist()):result[i]=value
            # Release this diagnostic graph before constructing the next one.
            # Unlike PPO, this read-only grad-mode probe never calls backward.
            del vd,td
    return result
def joint(p,prepared,reference=False):
    import torch
    result=[]
    with torch.no_grad():
        for row in prepared:
            with p._scope([row],reference):
                vd=p.verb_distribution([row],reference);parts=[]
                for v,branch in enumerate(row['branches']):
                    if branch:
                        td=p.target_distribution([row],[v],reference)
                        parts.extend((vd.probs[0,v]*td.probs[0,:len(branch)]).tolist())
            if abs(sum(parts)-1)>1e-5:raise ValueError('Joint probability not normalized')
            result.append(parts)
    return result
def controlled_states(c):
    from agent.m8s3_env import GodotEnv
    from agent.m8s3_contract import encode
    fixed=[]
    for seed in c.seeds:
        env=GodotEnv(c.godot,c.project)
        try:
            s=env.reset(seed,c.profile)
            for step in range(8):
                fixed.append(s['observation']);r=encode(s['observation']);branch=r['branches'][0]
                s=env.step(branch[step%len(branch)] if branch else r['branches'][5][0])
        finally:env.close()
    return fixed
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();reg=json.loads(Path(a.registration).read_text());verify(reg);c=SimpleNamespace(**reg['config'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy')
    setup();import torch
    from agent.m8s3_env import GodotEnv
    from agent.m8s3_contract import SurvivalReward
    from agent.m8x_train import finish_trajectory
    from agent.m8_train import update
    class CountAdamW(torch.optim.AdamW):
        steps=0
        def step(self,*args,**kw):
            result=super().step(*args,**kw);self.steps+=1;return result
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>c.wall_clock_limit_seconds:raise TimeoutError('Bounded diagnostic deadline')
    def timed(fn):
        torch.cuda.synchronize();t=time.perf_counter();r=fn();torch.cuda.synchronize();return r,time.perf_counter()-t
    fixed=controlled_states(c);write(out/'controlled_states.json',fixed)
    comparisons={};measurements={};baseline_prob=None;initial_critic=None
    write(out/'manifest.json',dict(registration_sha256=reg['registration_sha256'],config=reg['config'],formal_training_started=False,
        checkpoint_writes=0,timing_scope='3 x (128 fresh real decisions + all128 logp check + 2 PPO epochs + 8 critic epochs); loading/fixed checks/cache refresh checks excluded'))
    for mode in c.arms:
        deadline();p=make(c,mode);initial=p.digest();critic=critic_digest(p)
        if initial_critic is None:initial_critic=critic
        assert critic==initial_critic
        write(out/(mode+'_policy.json'),p.metadata());emit(event='arm_loaded',mode=mode,actor=initial,critic=critic)
        prepared=p.prepare(fixed);verbs=[];targets=[]
        for i,r in enumerate(prepared):
            legal=[v for v,b in enumerate(r['branches']) if b];v=legal[i%len(legal)];verbs.append(v);targets.append(i%len(r['branches'][v]))
        def fixed_forward():
            with torch.no_grad():
                for start in range(0,32,4):
                    batch=prepared[start:start+4];v=verbs[start:start+4]
                    p.distributions(batch,v);p.distributions(batch,v,True);p.values(batch)
        _,warm=timed(fixed_forward);times=[timed(fixed_forward)[1] for _ in range(3)]
        tables={'actor':joint(p,prepared),'reference':joint(p,prepared,True)}
        if baseline_prob is None:baseline_prob=tables
        old=selected_logps(p,prepared,verbs,targets);errors=[]
        for size in (1,3,4):
            values=selected_logps(p,prepared,verbs,targets,size,list(reversed(range(32))))
            errors.append(max(abs(x-y) for x,y in zip(old,values)))
        grad=selected_logps(p,prepared,verbs,targets,grad=True)
        numeric=dict(reorder_max=max(errors),grad_mode_max=max(abs(x-y) for x,y in zip(old,grad)))
        measurements[mode]=dict(warmup_seconds=warm,fixed_forward_seconds=times,**numeric,
            actor_vs_baseline=difference(baseline_prob['actor'],tables['actor']),reference_vs_baseline=difference(baseline_prob['reference'],tables['reference']))
        write(out/(mode+'_probabilities.json'),tables);write(out/'performance.json',measurements);emit(event='probability_checks',mode=mode,**measurements[mode])
        assert max(numeric.values())<=1e-6 and p.digest()==initial
        del p,prepared;gc.collect();torch.cuda.empty_cache()
        p=make(c,mode);assert critic_digest(p)==initial_critic and p.digest()==initial
        ref_initial=p.reference_digest()
        actor_opt=CountAdamW([q for q in p.model.parameters() if q.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0.)
        critic_opt=CountAdamW(p.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0.)
        assert not actor_opt.state and not critic_opt.state
        random.seed(c.rng_seed);torch.manual_seed(c.rng_seed)
        envs=[];states=[];trackers=[];reports=[];seeds=list(c.seeds);resets=[0]*4;action_counts=Counter()
        try:
            for seed in seeds:
                env=GodotEnv(c.godot,c.project);envs.append(env);s=env.reset(seed,c.profile);states.append(s);trackers.append(SurvivalReward(s['observation']))
            with (out/(mode+'_transitions.jsonl')).open('x',encoding='utf-8') as trace:
                for number in range(3):
                    deadline();trajectories=[[] for _ in envs];ticks=0
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    for step in range(32):
                        prepared=p.prepare([s['observation'] for s in states]);sampled=p.act(prepared,with_reference=True)
                        for i,env in enumerate(envs):
                            action=sampled['actions'][i];nxt=env.step(action);scored=trackers[i].score(nxt['transition'],nxt['observation'])
                            terminal=nxt['terminated'];v=sampled['verbs'][i];t=sampled['targets'][i]
                            row=dict(prepared=prepared[i],verb=v,target=t,value=sampled['values'][i],logp=sampled['logp'][i],verb_logp=sampled['verb_logp'][i],
                                reference_verb=sampled['reference_verb'][i],reference_target=sampled['reference_target'][i],reward=scored['reward'],
                                duration=scored['elapsed_ticks'],terminal=terminal,boundary=terminal,next_obs=nxt['observation'])
                            trajectories[i].append(row);ticks+=row['duration'];action_counts[action['verb']]+=1
                            record=dict(mode=mode,update=number+1,step=step,env=i,seed=seeds[i],source='model',observation=states[i]['observation'],
                                model_input=prepared[i]['messages'],verb_index=v,target_index=t,semantic_target=prepared[i]['target_texts'][v][t],
                                executed_verb_token_ids=prepared[i]['input_ids'],action=action,transition=nxt['transition'],
                                next_observation=nxt['observation'],reward=scored,terminal=terminal,log_probability=row['logp'],value=row['value'])
                            trace.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                            if terminal:
                                resets[i]+=1;seeds[i]=c.seeds[i]+4*resets[i];states[i]=env.reset(seeds[i],c.profile);trackers[i]=SurvivalReward(states[i]['observation'])
                            else:states[i]=nxt
                    rows=[]
                    for trajectory in trajectories:
                        with torch.no_grad():value=0. if trajectory[-1]['terminal'] else float(p.values(p.prepare([trajectory[-1]['next_obs']]))[0])
                        rows.extend(finish_trajectory(trajectory,value,c.gamma_tick,c.gae_lambda))
                    torch.cuda.synchronize();rollout=time.perf_counter()-start
                    previous_actor,previous_critic=actor_opt.steps,critic_opt.steps
                    def checked_update():
                        lp=selected_logps(p,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])
                        error=max(abs(x-r['logp']) for x,r in zip(lp,rows))
                        if error>1e-6:raise RuntimeError('All-row pre-update logp drift: '+str(error))
                        return dict(pre_update_all_logp_max=error,**update(p,actor_opt,critic_opt,rows,c))
                    metrics,elapsed=timed(checked_update)
                    assert actor_opt.steps-previous_actor==metrics['optimizer_steps']
                    finite=all(math.isfinite(x) for x in metrics.values() if type(x) in (float,int))
                    refresh_error=0.
                    if mode in ('baseline','optimized'):
                        check=[r['prepared'] for r in rows[:4]];vs=[r['verb'] for r in rows[:4]]
                        with torch.no_grad():
                            vd,td=p.distributions(check,vs);before=(vd.probs.clone(),td.probs.clone());p.execution_cache.revisions.clear();p.clear_prefix_cache()
                            for module in p.model.modules():
                                if hasattr(module,'_versioned_merged_weight'):module._versioned_merged_weight.revision=None
                                if hasattr(module,'_grouped_projection'):module._grouped_projection.group.revision=None
                            vd,td=p.distributions(check,vs);refresh_error=max(float((vd.probs-before[0]).abs().max()),float((td.probs-before[1]).abs().max()))
                    report=dict(update=number+1,decisions=len(rows),world_ticks=ticks,rollout_seconds=rollout,update_seconds=elapsed,finite=finite,
                        refresh_error=refresh_error,critic_optimizer_steps=critic_opt.steps-previous_critic,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        cache_stats=dict(p.cache_stats),**metrics)
                    reports.append(report);write(out/(mode+'_stability.json'),reports);emit(event='update',mode=mode,**report)
            p.assert_reference_frozen();comparisons[mode]=dict(updates=reports,initial_actor=initial,initial_critic=initial_critic,
                actor_changed=p.digest()!=initial,critic_changed=critic_digest(p)!=initial_critic,optimizers_initially_empty=True,
                actual_actor_steps=actor_opt.steps,actual_critic_steps=critic_opt.steps,reference_unchanged=p.reference_digest()==ref_initial,
                action_counts=dict(action_counts),natural_deaths=sum(resets))
        finally:
            for env in envs:env.close()
        del p,actor_opt,critic_opt,rows,trajectories,prepared;gc.collect();torch.cuda.empty_cache()
    gate=speed_gate(comparisons['baseline']['updates'],comparisons['optimized']['updates']);verify(reg)
    result=dict(registration_sha256=reg['registration_sha256'],performance=measurements,stability=comparisons,speed_gate=gate,
        probability_consistency_passed=all(x['reorder_max']<=1e-6 and x['grad_mode_max']<=1e-6 for x in measurements.values()),
        sources_and_inputs_unchanged=True,formal_training_started=False,checkpoint_writes=0,seconds=time.monotonic()-started)
    write(out/'comparison.json',result);emit(event='completed',speed_gate=gate)
if __name__=='__main__':main()
