"""Registered end-to-end speed validation, without formal training/checkpoints."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def speed_gate(baseline,candidate):
    equal=(len(baseline)==len(candidate)==3 and
           all(a['decisions']==b['decisions']==128 and a['optimizer_steps']==b['optimizer_steps']==8
               for a,b in zip(baseline,candidate)))
    valid=all(r['finite'] and r['zero_update_logp_max']<=1e-6 for r in baseline+candidate)
    bt=[r['rollout_seconds']+r['update_seconds'] for r in baseline]
    ct=[r['rollout_seconds']+r['update_seconds'] for r in candidate]
    if not ct or min(ct)<=0:return dict(passed=False,equal_work=False)
    median=statistics.median(bt)/statistics.median(ct)
    total=sum(bt)/sum(ct)
    return dict(passed=equal and valid and median>2. and total>2.,equal_work=equal,
                numeric_checks=valid,median_speedup=median,all_three_rounds_speedup=total,
                baseline_round_seconds=bt,candidate_round_seconds=ct)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; existing jobs untouched')
    import torch
    from agent.m8x_kv2_registration import verify,sha
    from agent.m8x_policy import Policy
    from agent.m8x_fast_policy import FastPolicy
    from agent.m8x_kv_policy import KVPolicy
    from agent.m8x_causal_policy import CausalKVPolicy
    from agent.m8x_merged_lora import enable_merged_lora
    from agent.m8x_fused_norm import enable_fused_norm
    from agent.m8x_grouped_lora import GroupedPolicy as OptimizedPolicy
    from agent.m8x_env import GodotEnv
    from agent.m8x_contract import SurvivalReward,target_messages,VERSION
    from agent.m8x_train import finish_trajectory
    from agent.m8_train import update
    reg=json.loads(Path(args.registration).read_text())
    c=SimpleNamespace(**reg['config']);perf=c.performance
    verify(c.project,reg)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    dataset=[json.loads(x) for x in Path(args.dataset).read_text().splitlines()]
    fixed=[dataset[i*(len(dataset)-1)//63] for i in range(64)]
    manifest=dict(registration_sha256=reg['registration_sha256'],dataset_sha256=sha(args.dataset),
        formal_training_started=False,checkpoints_saved=0,config=reg['config'],
        timing_repeats=3,probability_states=64,
        cold_costs_in_primary_rounds=True,model_loading_and_development_warmup_separate=True)
    write(output/'manifest.json',manifest)
    began=time.monotonic()
    def deadline():
        if time.monotonic()-began>perf['wall_clock_limit_seconds']:raise TimeoutError('Registered benchmark budget exhausted')
    def emit(value):print(json.dumps(value,allow_nan=False),flush=True)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,time.perf_counter()-start
    def make(mode):
        options={}
        if mode=='baseline':cls=Policy
        elif mode=='compact':cls=FastPolicy;options['optimization']='compact'
        elif mode=='merged_compact':cls=FastPolicy;options['optimization']='compact'
        elif mode=='merged_kv':cls=KVPolicy
        elif mode=='merged_causal':cls=CausalKVPolicy
        elif mode=='optimized':cls=OptimizedPolicy;options.update(registration=reg,fused_norm=perf.get('fused_rms_norm',True))
        else:raise ValueError(mode)
        p=cls(c.model,checkpoint=c.initial_checkpoint,parent_checkpoint=c.parent_checkpoint,
              max_tokens=c.max_tokens,seed=perf['rng_seed'],with_reference=True,**options)
        if mode.startswith('merged_'):
            enable_merged_lora(p.model);enable_merged_lora(p.reference_model)
            if perf.get('fused_rms_norm',False):
                enable_fused_norm(p.model);enable_fused_norm(p.reference_model)
        return p
    @torch.no_grad()
    def joint(p,prepared,ref):
        result=[]
        for start in range(0,len(prepared),4):
            batch=prepared[start:start+4];vd=p.verb_distribution(batch,ref);tables=[[] for _ in batch]
            for verb in range(8):
                active=[i for i,row in enumerate(batch) if row['branches'][verb]]
                if not active:continue
                td=p.target_distribution([batch[i] for i in active],[verb]*len(active),ref)
                for j,i in enumerate(active):tables[i].extend((vd.probs[i,verb]*td.probs[j,:len(batch[i]['branches'][verb])]).tolist())
            result.extend(tables)
        return result
    def probability_difference(base,new):
        kls=[];diff=[];changed=0
        for a,b in zip(base,new):
            assert len(a)==len(b) and abs(sum(b)-1)<1e-5
            diff.append(max(abs(x-y) for x,y in zip(a,b)))
            kls.append(sum(x*math.log(max(x,1e-30)/max(y,1e-30)) for x,y in zip(a,b)))
            changed+=max(range(len(a)),key=a.__getitem__)!=max(range(len(b)),key=b.__getitem__)
        return dict(max_probability_difference=max(diff),mean_kl=statistics.mean(kls),max_kl=max(kls),greedy_changes=changed)
    measurements={};stability={};base_probability=None
    for mode in perf['arms']:
        deadline();p=make(mode);prepared=p.prepare([r['observation'] for r in fixed]);initial=p.digest()
        metadata=p.metadata()
        if mode=='optimized':assert metadata['inference_layout']==c.policy_layout and metadata['optimization_registered']
        write(output/(mode+'_policy.json'),metadata)
        def fixed_forward():
            with torch.no_grad():
                for start in range(0,64,4):
                    batch=prepared[start:start+4];verbs=[r['verb_index'] for r in fixed[start:start+4]]
                    p.distributions(batch,verbs);p.distributions(batch,verbs,True);p.values(batch)
        _,warm=timed(fixed_forward)
        timings=[timed(fixed_forward)[1] for _ in range(3)]
        probabilities={'actor':joint(p,prepared,False),'reference':joint(p,prepared,True)}
        if base_probability is None:base_probability=probabilities
        original=[]
        with torch.no_grad():
            for start in range(0,64,4):
                v=torch.tensor([r['verb_index'] for r in fixed[start:start+4]],device='cuda')
                t=torch.tensor([r['target_index'] for r in fixed[start:start+4]],device='cuda')
                a,b=p.distributions(prepared[start:start+4],v.tolist())
                original.extend((a.log_prob(v)+b.log_prob(t)).tolist())
        reorder=grad_error=0.
        order=list(reversed(range(64)))
        for size in (1,3,4):
            with torch.no_grad():
                for start in range(0,64,size):
                    ix=order[start:start+size]
                    v=torch.tensor([fixed[i]['verb_index'] for i in ix],device='cuda')
                    t=torch.tensor([fixed[i]['target_index'] for i in ix],device='cuda')
                    a,b=p.distributions([prepared[i] for i in ix],v.tolist())
                    values=(a.log_prob(v)+b.log_prob(t)).tolist()
                    reorder=max(reorder,max(abs(values[j]-original[i]) for j,i in enumerate(ix)))
        for start in range(0,16,4):
            v=torch.tensor([r['verb_index'] for r in fixed[start:start+4]],device='cuda')
            t=torch.tensor([r['target_index'] for r in fixed[start:start+4]],device='cuda')
            a,b=p.distributions(prepared[start:start+4],v.tolist())
            values=(a.log_prob(v)+b.log_prob(t)).detach().tolist()
            grad_error=max(grad_error,max(abs(x-original[start+j]) for j,x in enumerate(values)))
            del a,b
        assert p.digest()==initial
        assert reorder<=1e-6 and grad_error<=1e-6
        measurements[mode]=dict(warmup_seconds=warm,fixed_forward_seconds=timings,
            zero_update_reordered_max=reorder,zero_update_grad_mode_max=grad_error,
            actor_vs_baseline=probability_difference(base_probability['actor'],probabilities['actor']),
            reference_vs_baseline=probability_difference(base_probability['reference'],probabilities['reference']))
        write(output/(mode+'_probabilities.json'),probabilities)
        write(output/'performance.json',measurements);emit(dict(event='probabilities_complete',mode=mode,**measurements[mode]))
        # Start every stability arm again from the identical initial checkpoint.
        del p,prepared;gc.collect();torch.cuda.empty_cache();p=make(mode)
        actor_initial=p.digest();reference_initial=p.reference_digest()
        actor_opt=torch.optim.AdamW([q for q in p.model.parameters() if q.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0.)
        critic_opt=torch.optim.AdamW(p.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0.)
        random.seed(perf['rng_seed']);torch.manual_seed(perf['rng_seed'])
        envs=[];states=[];trackers=[];reports=[];seeds=list(perf['seeds']);resets=[0]*4
        try:
            for seed in perf['seeds']:
                env=GodotEnv(c.godot,c.project);envs.append(env);state=env.reset(seed,'open3')
                states.append(state);trackers.append(SurvivalReward(state['observation']))
            with (output/(mode+'_transitions.jsonl')).open('x',encoding='utf-8') as trace:
                for number in range(3):
                    deadline();trajectories=[[] for _ in envs]
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                    for step in range(32):
                        prepared=p.prepare([s['observation'] for s in states]);sampled=p.act(prepared,with_reference=True)
                        for i,env in enumerate(envs):
                            action=sampled['actions'][i];nxt=env.step(action)
                            scored=trackers[i].score(nxt['transition'],nxt['observation'])
                            terminal=nxt['terminated']
                            v=sampled['verbs'][i];t=sampled['targets'][i]
                            row=dict(prepared=prepared[i],verb=v,target=t,value=sampled['values'][i],
                                logp=sampled['logp'][i],verb_logp=sampled['verb_logp'][i],reference_verb=sampled['reference_verb'][i],
                                reference_target=sampled['reference_target'][i],reward=scored['reward'],duration=scored['elapsed_ticks'],
                                terminal=terminal,boundary=terminal,next_obs=nxt['observation'])
                            trajectories[i].append(row)
                            record=dict(mode=mode,update=number+1,step=step,env=i,seed=seeds[i],source='model',salt='m6',contract=VERSION,
                                observation=states[i]['observation'],model_input=prepared[i]['messages'],verb_index=v,target_index=t,
                                target_input=target_messages(prepared[i],v) if len(prepared[i]['branches'][v])>1 else None,
                                executed_verb_token_ids=prepared[i]['input_ids'],
                                executed_target_token_ids=p._target_tokens(prepared[i],v) if len(prepared[i]['branches'][v])>1 else None,
                                action=action,transition=nxt['transition'],next_observation=nxt['observation'],reward=scored,
                                terminal=terminal,truncated=False,log_probability=row['logp'],value=row['value'])
                            trace.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                            if terminal:
                                resets[i]+=1;seeds[i]=perf['seeds'][i]+4*resets[i]
                                states[i]=env.reset(seeds[i],'open3');trackers[i]=SurvivalReward(states[i]['observation'])
                            else:states[i]=nxt
                    rows=[]
                    for trajectory in trajectories:
                        with torch.no_grad():value=0. if trajectory[-1]['terminal'] else float(p.values(p.prepare([trajectory[-1]['next_obs']]))[0])
                        rows.extend(finish_trajectory(trajectory,value,c.gamma_tick,c.gae_lambda))
                    torch.cuda.synchronize();rollout=time.perf_counter()-start
                    metrics,seconds=timed(lambda:update(p,actor_opt,critic_opt,rows,c))
                    finite=all(math.isfinite(v) for v in metrics.values() if type(v) in (float,int))
                    cache_error=0.
                    if mode.startswith('merged_') or mode=='optimized':
                        check=[r['prepared'] for r in rows[:4]];verbs=[r['verb'] for r in rows[:4]]
                        with torch.no_grad():
                            a,b=p.distributions(check,verbs);before=(a.probs.clone(),b.probs.clone())
                            for module in p.model.modules():
                                if hasattr(module,'_versioned_merged_weight'):module._versioned_merged_weight.revision=None
                            p.execution_cache.revisions.clear()
                            for module in p.model.modules():
                                if hasattr(module,'_grouped_projection'):
                                    assert module._grouped_projection.pending is None
                                    module._grouped_projection.group.revision=None
                            a,b=p.distributions(check,verbs)
                            cache_error=max(float((a.probs-before[0]).abs().max()),float((b.probs-before[1]).abs().max()))
                        assert cache_error<=1e-6 and getattr(p,'_kv_scope',None) is None
                    report=dict(update=number+1,decisions=128,rollout_seconds=rollout,update_seconds=seconds,
                                finite=finite,weight_cache_refresh_max=cache_error,kv_scope_released=getattr(p,'_kv_scope',None) is None,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),**metrics)
                    reports.append(report);write(output/(mode+'_stability.json'),reports)
                    emit(dict(event='update',mode=mode,**report))
            p.assert_reference_frozen()
            stability[mode]=dict(updates=reports,actor_changed=p.digest()!=actor_initial,
                                 reference_unchanged=p.reference_digest()==reference_initial)
        finally:
            for env in envs:env.close()
        del p,actor_opt,critic_opt,prepared,rows,trajectories;gc.collect();torch.cuda.empty_cache()
    candidate=perf['arms'][-1]
    gate=speed_gate(stability['baseline']['updates'],stability[candidate]['updates'])
    verify(c.project,reg)
    gates=dict(speed_over_two=gate['passed'],probability_recomputation=all(m['zero_update_reordered_max']<=1e-6 and m['zero_update_grad_mode_max']<=1e-6 for m in measurements.values()),
        frozen_reference=all(a['reference_unchanged'] for a in stability.values()),
        real_actor_updates=all(a['actor_changed'] for a in stability.values()),registration_unchanged=True)
    result=dict(manifest=manifest,performance=measurements,stability=stability,speed_gate=gate,gates=gates,
                passed=all(gates.values()),seconds=time.monotonic()-began,formal_training_started=False,checkpoints_saved=0)
    write(output/'comparison.json',result);emit(dict(event='completed',passed=result['passed'],speed_gate=gate,gates=gates))
    if not result['passed']:raise RuntimeError('Registered optimization gate did not pass')


if __name__=='__main__':main()
