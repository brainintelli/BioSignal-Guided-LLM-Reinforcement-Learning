"""Bounded node-5 layout comparison; never calls the formal trainer or saves weights.

Every mode is compared on identical immutable real observations. Each short
stability arm starts from the same checkpoint and collects its own on-policy
Godot transitions; no old-layout probability enters a new-layout PPO ratio.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU occupied; existing work left untouched: ' + active)
    import torch
    from agent.m8x_policy import Policy
    from agent.m8x_fast_policy import FastPolicy, parameter_revision
    from agent.m8x_env import GodotEnv
    from agent.m8x_contract import SurvivalReward, target_messages, VERSION
    from agent.m8x_train import finish_trajectory
    from agent.m8_train import update

    torch.set_num_threads(8)
    root = Path('/home/researcher/meadowmind-gpu')
    project = root/'m8b-project'
    c = SimpleNamespace(**json.loads((project/'configs/m8x_exploration_attempt1.json').read_text()))
    checkpoint = root/'m8b-runs/exploration_attempt1/checkpoint_0032'
    records = [json.loads(x) for x in Path(args.dataset).read_text().splitlines()]
    # Exactly 64 records spread across the whole previous real diagnostic.
    indices = [i*(len(records)-1)//63 for i in range(64)]
    records = [records[i] for i in indices]
    registration = json.loads((checkpoint.parent/'registration.json').read_text())
    checked_files = {str(project/rel): digest for rel,digest in registration['source_hashes'].items()}
    for directory in [checkpoint, root/'m6-runs/small_v2_stable/checkpoint_0016']:
        for pattern in ['adapter/*.safetensors','adapter/*.json','critic.pt','policy.json']:
            for path in directory.glob(pattern):
                checked_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), formal_training=False,
                    dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                    dataset_record_indices=indices, timing_repeats=3, states=64,
                    modes=['baseline','exact','compact','prefix'],
                    stability_modes=['baseline','compact','prefix'],
                    stability_updates=3, decisions_per_update=64, max_optimizer_steps_per_arm=12,
                    stability_world_seeds=[882710,882711,882712,882713], sampling_seed=20261108,
                    required_zero_update_logp_tolerance=1e-6, exact_baseline_probability_tolerance=1e-6,
                    approximate_layout_is_drop_in_equivalent=False, checkpoint_saved=False,
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    source_hashes={rel:hashlib.sha256((project/rel).read_bytes()).hexdigest()
                                   for rel in ['agent/m8x_fast_layout.py','agent/m8x_fast_policy.py',
                                               'tools/benchmark_m8x_fast_policy.py']})
    write(output/'diagnostic_manifest.json',manifest)
    began = time.monotonic()
    def deadline():
        if time.monotonic()-began > 1500:
            raise TimeoutError('Bounded diagnostic exceeded 25 minutes')

    def make(mode):
        kwargs=dict(checkpoint=str(checkpoint), parent_checkpoint=c.parent_checkpoint,
                    max_tokens=c.max_tokens,seed=20261108,with_reference=True)
        return Policy(c.model,**kwargs) if mode=='baseline' else FastPolicy(c.model,optimization=mode,**kwargs)

    def elapsed(fn):
        torch.cuda.synchronize(); start=time.perf_counter()
        result=fn()
        torch.cuda.synchronize()
        return result,time.perf_counter()-start

    @torch.no_grad()
    def joint_probabilities(policy, prepared, reference=False):
        result=[]
        for start in range(0,len(prepared),4):
            batch=prepared[start:start+4]
            vd=policy.verb_distribution(batch,reference)
            tables=[[] for _ in batch]
            for verb in range(8):
                active=[i for i,row in enumerate(batch) if row['branches'][verb]]
                if not active:
                    continue
                td=policy.target_distribution([batch[i] for i in active],[verb]*len(active),reference)
                for j,i in enumerate(active):
                    tables[i].extend((vd.probs[i,verb]*td.probs[j,:len(batch[i]['branches'][verb])]).tolist())
            result.extend(tables)
        return result

    def compare_probabilities(baseline, candidate):
        diffs=[]; kls=[]; tvs=[]; changes=0
        for b,n in zip(baseline,candidate):
            assert len(b)==len(n) and abs(sum(n)-1.)<1e-5
            diffs.append(max(abs(x-y) for x,y in zip(b,n)))
            kls.append(sum(x*math.log(max(x,1e-30)/max(y,1e-30)) for x,y in zip(b,n)))
            tvs.append(.5*sum(abs(x-y) for x,y in zip(b,n)))
            changes += max(range(len(b)),key=b.__getitem__)!=max(range(len(n)),key=n.__getitem__)
        return dict(max_action_probability_difference=max(diffs), mean_joint_kl=statistics.mean(kls),
                    max_joint_kl=max(kls),mean_total_variation=statistics.mean(tvs),
                    max_total_variation=max(tvs),greedy_action_changed_states=changes,states=len(baseline))

    results={}
    base_prob=base_ref=None
    for mode in manifest['modes']:
        deadline(); policy=make(mode); prepared=policy.prepare([r['observation'] for r in records])
        digest=policy.digest()
        # Pre-warm every measured shape, including target branches selected in the fixed dataset.
        def fixed_workload():
            for start in range(0,len(prepared),4):
                batch=prepared[start:start+4]
                verbs=[r['verb_index'] for r in records[start:start+4]]
                with torch.no_grad():
                    policy.distributions(batch,verbs)
                    policy.distributions(batch,verbs,True)
                    policy.values(batch)
        _,warm=elapsed(fixed_workload)
        emit(dict(event='warmup_complete',mode=mode,seconds=warm))
        samples=[elapsed(fixed_workload)[1] for _ in range(3)]
        # Separate actual act() timing so categorical sampling is included as well.
        act_samples=[]
        for _ in range(3):
            torch.manual_seed(20261108)
            _,seconds=elapsed(lambda:[policy.act(prepared[i:i+4],with_reference=True)
                                      for i in range(0,len(prepared),4)])
            act_samples.append(seconds)
        probabilities=joint_probabilities(policy,prepared)
        reference=joint_probabilities(policy,prepared,True)
        if base_prob is None:
            base_prob,base_ref=probabilities,reference
        # Probe reshuffled/minibatch/grad-mode invariance on all 64 recorded decisions.
        original=[]
        with torch.no_grad():
            for start in range(0,len(prepared),4):
                local=records[start:start+4];v=torch.tensor([r['verb_index'] for r in local],device='cuda')
                t=torch.tensor([r['target_index'] for r in local],device='cuda')
                vd,td=policy.distributions(prepared[start:start+4],v.tolist())
                original.extend((vd.log_prob(v)+td.log_prob(t)).tolist())
        largest=0.
        order=list(reversed(range(len(prepared))))
        for size in [1,3,4]:
            with torch.no_grad():
                for start in range(0,len(order),size):
                    ix=order[start:start+size]
                    v=torch.tensor([records[i]['verb_index'] for i in ix],device='cuda')
                    t=torch.tensor([records[i]['target_index'] for i in ix],device='cuda')
                    vd,td=policy.distributions([prepared[i] for i in ix],v.tolist())
                    now=(vd.log_prob(v)+td.log_prob(t)).tolist()
                    largest=max(largest,max(abs(now[j]-original[i]) for j,i in enumerate(ix)))
        grad_mode_error=0.
        for start in range(0,16,4):
            local=records[start:start+4]
            v=torch.tensor([r['verb_index'] for r in local],device='cuda')
            t=torch.tensor([r['target_index'] for r in local],device='cuda')
            vd,td=policy.distributions(prepared[start:start+4],v.tolist())
            now=(vd.log_prob(v)+td.log_prob(t)).detach().tolist()
            grad_mode_error=max(grad_mode_error,max(abs(x-original[start+j]) for j,x in enumerate(now)))
            # Release this graph BEFORE constructing the next two-stage graph.
            del vd,td
        assert policy.digest()==digest
        case=dict(mode=mode,warmup_seconds=warm,fixed_workload_seconds=samples,
                  fixed_workload_median_seconds=statistics.median(samples),
                  act_seconds=act_samples,act_median_seconds=statistics.median(act_samples),
                  act_decisions_per_second=64/statistics.median(act_samples),
                  actor_vs_baseline=compare_probabilities(base_prob,probabilities),
                  reference_vs_baseline=compare_probabilities(base_ref,reference),
                  reordered_zero_update_logp_max=largest,grad_mode_logp_max=grad_mode_error,
                  actor_unchanged=True,prefix_tokens=getattr(policy,'prefix_ids',[]).__len__())
        results[mode]=case
        write(output/(mode+'_probabilities.json'),dict(actor=probabilities,reference=reference))
        write(output/'performance.json',results);emit(dict(event='performance_complete',**case))
        del policy,prepared;gc.collect();torch.cuda.empty_cache()

    stability={}
    for mode in manifest['stability_modes']:
        deadline();policy=make(mode)
        initial=policy.digest();reference_initial=policy.reference_digest()
        actor_opt=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=c.actor_lr,eps=1e-5,weight_decay=0.)
        critic_opt=torch.optim.AdamW(policy.critic.parameters(),lr=c.critic_lr,eps=1e-5,weight_decay=0.)
        envs=[];states=[];trackers=[];reports=[]
        random.seed(20261108);torch.manual_seed(20261108)
        records_path=output/(mode+'_fresh_transitions.jsonl')
        try:
            for seed in manifest['stability_world_seeds']:
                env=GodotEnv(c.godot,c.project);envs.append(env)
                state=env.reset(seed,'open3');states.append(state);trackers.append(SurvivalReward(state['observation']))
            with records_path.open('x',encoding='utf-8') as trace:
                for number in range(3):
                    deadline();trajectories=[[] for _ in envs]
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
                    for step in range(16):
                        prepared=policy.prepare([s['observation'] for s in states])
                        sampled=policy.act(prepared,with_reference=True)
                        for i,env in enumerate(envs):
                            action=sampled['actions'][i];nxt=env.step(action)
                            scored=trackers[i].score(nxt['transition'],nxt['observation'])
                            if nxt['terminated']:
                                raise RuntimeError('Unexpected short-probe death: no synthetic reset is allowed')
                            row=dict(prepared=prepared[i],verb=sampled['verbs'][i],target=sampled['targets'][i],
                                     value=sampled['values'][i],logp=sampled['logp'][i],verb_logp=sampled['verb_logp'][i],
                                     reference_verb=sampled['reference_verb'][i],reference_target=sampled['reference_target'][i],
                                     reward=scored['reward'],duration=scored['elapsed_ticks'],terminal=False,
                                     boundary=False,next_obs=nxt['observation'])
                            trajectories[i].append(row)
                            record=dict(mode=mode,update=number+1,step=step,env=i,seed=manifest['stability_world_seeds'][i],
                                        observation=states[i]['observation'],model_input=prepared[i]['messages'],
                                        verb_index=row['verb'],target_index=row['target'],
                                        target_input=target_messages(prepared[i],row['verb']) if len(prepared[i]['branches'][row['verb']])>1 else None,
                                        action=action,transition=nxt['transition'],next_observation=nxt['observation'],
                                        reward=scored,terminal=False,truncated=False,source='model',salt='m6',contract=VERSION,
                                        log_probability=row['logp'],value=row['value'])
                            trace.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                            states[i]=nxt
                    rows=[]
                    for trajectory in trajectories:
                        with torch.no_grad():value=float(policy.values(policy.prepare([trajectory[-1]['next_obs']]))[0])
                        rows.extend(finish_trajectory(trajectory,value,c.gamma_tick,c.gae_lambda))
                    torch.cuda.synchronize();rollout_seconds=time.perf_counter()-begin
                    metrics,update_seconds=elapsed(lambda:update(policy,actor_opt,critic_opt,rows,c))
                    finite=all(math.isfinite(v) for v in metrics.values() if type(v) in (float,int))
                    cache_error=0.;cache_revision_current=True
                    if mode=='prefix':
                        check=[r['prepared'] for r in rows[:4]];verbs=[r['verb'] for r in rows[:4]]
                        with torch.no_grad():
                            a,b=policy.distributions(check,verbs)
                            warm=(a.probs.clone(),b.probs.clone())
                            cache_revision_current=policy._inference_prefixes[False][0]==parameter_revision(policy.model)
                            policy._inference_prefixes.clear()
                            a,b=policy.distributions(check,verbs)
                            cache_error=max(float((a.probs-warm[0]).abs().max()),float((b.probs-warm[1]).abs().max()))
                    report=dict(update=number+1,decisions=64,rollout_seconds=rollout_seconds,update_seconds=update_seconds,
                                finite=finite,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                                cache_refresh_probability_error=cache_error,cache_revision_current=cache_revision_current,**metrics)
                    reports.append(report);write(output/(mode+'_stability.json'),reports)
                    emit(dict(event='stability_update',mode=mode,**report))
            policy.assert_reference_frozen()
            stability[mode]=dict(updates=reports,actor_changed=policy.digest()!=initial,
                                 reference_unchanged=policy.reference_digest()==reference_initial,
                                 optimizer_steps=sum(r['optimizer_steps'] for r in reports),
                                 real_decisions=192,checkpoints_saved=0)
        finally:
            for env in envs:env.close()
        del actor_opt,critic_opt,policy,prepared,rows,trajectories
        gc.collect();torch.cuda.empty_cache()
    source_and_weight_mismatches=[path for path,digest in checked_files.items()
                                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest]
    gates=dict(
        exact_probability_match=results['exact']['actor_vs_baseline']['max_action_probability_difference']<=1e-6
                                and results['exact']['reference_vs_baseline']['max_action_probability_difference']<=1e-6,
        all_zero_update_probabilities=all(r['reordered_zero_update_logp_max']<=1e-6 and r['grad_mode_logp_max']<=1e-6 for r in results.values()),
        all_stability_finite=all(u['finite'] and u['zero_update_logp_max']<=1e-6 and u['cache_revision_current'] and u['cache_refresh_probability_error']<=1e-6
                                 for arm in stability.values() for u in arm['updates']),
        all_actors_updated=all(a['actor_changed'] and a['optimizer_steps']>0 for a in stability.values()),
        all_references_frozen=all(a['reference_unchanged'] for a in stability.values()),
        original_files_unchanged=not source_and_weight_mismatches)
    result=dict(manifest=manifest,performance=results,stability=stability,gates=gates,passed=all(gates.values()),
                original_files_checked=len(checked_files),source_and_weight_mismatches=source_and_weight_mismatches,
                seconds=time.monotonic()-began,formal_training_started=False,checkpoints_saved=0,
                finished_utc=datetime.now(timezone.utc).isoformat())
    write(output/'comparison.json',result);emit(dict(event='completed',passed=result['passed'],gates=gates,seconds=result['seconds']))
    if not result['passed']:raise RuntimeError('Optimization diagnostic gate failed; see comparison.json')


if __name__=='__main__':main()
