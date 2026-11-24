"""Eight native worlds and one synchronously trained R1-only policy."""
import argparse,hashlib,json,os,socket,subprocess,time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
# Select the worker's GPU before importing policy/kernel modules that initialize CUDA.
if 'LOCAL_RANK' in os.environ:torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
import common as c
from common import b
from protocol import optimizer_gate,resource_gate
from distributed_update import update,global_assert
from host_cache import release_host_cache

PROTOCOL=dict(world_size=8,rounds=8,local_decisions=16,global_decisions=128,global_minibatch=32,local_microbatch=4,
    world_seed=4156101,rng_seed=4166101,alpha=1.,r2_enabled=False,state2_enabled=False,stimulation=False,
    distribution='synchronous NCCL gradients; rank-stratified minibatches; rank-local trajectories',initial_checkpoint=b.CONFIG['warm_checkpoint'],
    host_cache_policy='release unused pinned-host cache after completed round; preserve before/after RSS')

def objects(value):
    result=[None]*dist.get_world_size();dist.all_gather_object(result,value);return result

def identical(value,label):
    all_values=objects(value);global_assert(all(v==all_values[0] for v in all_values),label+' differs across ranks');return all_values

def optimizer_tensor_hash(items):
    h=hashlib.sha256()
    for name,tensor in items:
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def optimizer_hash(opts):
    items=[]
    for name,opt in sorted(opts.items()):
        for i,entry in enumerate(opt.state.values()):
            for key,tensor in sorted(entry.items()):
                assert isinstance(tensor,torch.Tensor);items.append((f'{name}/{i}/{key}',tensor))
    return optimizer_tensor_hash(items)

def saved_optimizer_hash(path):
    value=torch.load(path,map_location='cpu',weights_only=False);items=[];steps={}
    for name in ['actor','critic']:
        entries=value[name]['state'].values();steps[name]=[int(s['step'].item()) for s in entries if 'step' in s]
        for i,entry in enumerate(value[name]['state'].values()):
            for key,tensor in sorted(entry.items()):items.append((f'{name}/{i}/{key}',tensor))
    return optimizer_tensor_hash(items),steps

def source_identity():return dict(**c.identity(['distributed_update.py','eightgpu_run.py','host_cache.py']),protocol=PROTOCOL)

def init():
    rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==8
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=240),device_id=torch.device('cuda',rank))
    assert rank==dist.get_rank() and torch.cuda.device_count()==8 and socket.gethostname()=='researcher-local-6'
    return rank

def collect(out,rank):
    if rank==0:
        # Both native-tail execution and independent verification must precede this run.
        tv=b.read(b.WORK/'tail01/independent_verification.json');assert tv['passed']
        c.check_manifest(b.WORK/'tail01','collection_manifest.json')
        assert tv['collection_manifest_sha256']==b.sha(b.WORK/'tail01/collection_manifest.json')
        assert b.read(b.WORK/'distributed_math.json')['passed']
        out.mkdir(exist_ok=False)
    dist.barrier();own=out/f'rank_{rank:02d}';own.mkdir(exist_ok=False);began=time.monotonic()
    src=source_identity();identical(src,'source/protocol');b.write(own/'identity.json',src)
    gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).splitlines()
    gpu=dict(rank=rank,device=torch.cuda.current_device(),uuid=gpu_lines[rank].split(',')[1].strip(),pid=os.getpid(),host=socket.gethostname())
    inventory=objects(gpu);assert len({x['uuid'] for x in inventory})==8 and len({x['pid'] for x in inventory})==8
    b.write(own/'device.json',gpu)
    if rank==0:b.write(out/'device_inventory.json',inventory)
    policy=b.new_policy();_,_,migration=b.initial_probe(policy);b.write(own/'migration_probe.json',migration)
    initial=c.state(policy);identical(initial,'original model');b.write(own/'initial_parameters.json',initial)
    opts=c.optimizers(policy);param_ids={k:[id(p) for g in opt.param_groups for p in g['params']] for k,opt in opts.items()}
    torch.manual_seed(PROTOCOL['rng_seed']+rank);seed=PROTOCOL['world_seed']+rank*10000;summaries=[]
    with c.environment() as env:
        obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
        for number in range(1,PROTOCOL['rounds']+1):
            rd=own/f'round_{number:02d}';rd.mkdir();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
            before=c.state(policy);before_steps=c.steps(opts);before_opt_hash=optimizer_hash(opts);identical(dict(model=before,steps=before_steps,optimizer=before_opt_hash),'pre-update state')
            rows=[];events=[]
            for i in range(PROTOCOL['local_decisions']):
                r,e,obs=c.sample(policy,env,tracker,obs,i,seed);rows.append(r);events.append(e)
                if r['terminal']:
                    seed+=1;obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
                    e['reset']=dict(seed=seed,observation=obs)
            boot=c.finish(policy,rows,obs);c.record(rd,rows,events,boot)
            pre=b.max_difference(c.statistics(policy,rows),b.row_statistics(rows));global_assert(pre<=1e-6,'rank-local pre-update statistic error')
            input_hash=b.digest([b.plain(r['prepared']) for r in rows])
            if rank==0:b.emit('eightgpu_update_started',round=number,global_decisions=128,old_statistic_errors=objects(pre))
            else:objects(pre)
            with b.saved_on_cpu_preserve_layout():metrics=update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**b.CONFIG))
            torch.cuda.synchronize();after=c.state(policy);after_steps=c.steps(opts);after_opt_hash=optimizer_hash(opts)
            optimizer_gate(before_steps,after_steps,metrics['optimizer_steps'],metrics['critic_optimizer_steps'])
            assert metrics['critic_optimizer_steps']==64
            identical(dict(model=after,steps=after_steps,optimizer=after_opt_hash,metrics=metrics),'joint update')
            checks=c.checks(policy,initial,before,after,rows,input_hash)
            checks['optimizer_objects_retained']=param_ids=={k:[id(p) for g in opt.param_groups for p in g['params']] for k,opt in opts.items()}
            global_assert(all(checks.values()),'post-update checks failed')
            post=c.statistics(policy,rows);b.write(rd/'post_update_statistics.json',post)
            checkpoint=out/f'checkpoint_{number:02d}'
            if rank==0:b.save(policy,checkpoint,opts['actor'],opts['critic'],number,b.digest(src))
            dist.barrier();b.load(policy,checkpoint)
            error=b.max_difference(c.statistics(policy,rows),post);global_assert(error<=1e-6,'same-rank checkpoint reload differs')
            assert c.steps(opts)==after_steps and optimizer_hash(opts)==after_opt_hash
            b.write(rd/'rng.json',dict(cpu=torch.get_rng_state().numpy().tobytes().hex(),cuda=torch.cuda.get_rng_state().cpu().numpy().tobytes().hex()))
            summary=dict(round=number,rank=rank,decisions=len(rows),ticks=sum(r['duration'] for r in rows),terminal_indices=[i for i,r in enumerate(rows) if r['terminal']],
                before=before,after=after,optimizer_steps_before=before_steps,optimizer_steps_after=after_steps,optimizer_hash_before=before_opt_hash,optimizer_hash_after=after_opt_hash,
                metrics=metrics,checks=checks,pre_error=pre,reload_error=error)
            del rows,events,post,r,e
            before_release=c.resources(policy);release_host_cache();summary['resources']=c.resources(policy)
            summary['resources']['rss_before_host_cache_release']=before_release['rss']
            summary['resources']['peak_rss_process']=int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmHWM:')))*1024
            assert summary['resources']['allocated']==before_release['allocated']
            summary['seconds']=time.monotonic()-begin
            b.write(rd/'report.json',summary);c.manifest(rd,'collection_manifest.json');summaries.append(summary)
            aggregate=objects(dict(rank=rank,seconds=summary['seconds'],resources=summary['resources'],pre_error=pre,reload_error=error,actor_steps=metrics['optimizer_steps']))
            if rank==0:b.emit('eightgpu_round_passed',round=number,ranks=aggregate)
    assert source_identity()==src
    screen=resource_gate([r['resources'] for r in summaries]);report=dict(passed=screen['passed'],rank=rank,collection_pid=os.getpid(),gpu=gpu,rounds=8,decisions=128,
        ticks=sum(s['ticks'] for s in summaries),terminals=sum(len(s['terminal_indices']) for s in summaries),seconds=time.monotonic()-began,
        final_actor=policy.digest(),resource_gate=screen,alpha=1.,r2_enabled=False,state2_enabled=False,stimulation=False,
        cross_host_probe_passed=migration['cross_host_passed'],cross_host_probe_error=migration['cross_host_error'])
    b.write(own/'report.json',report);c.manifest(own,'collection_manifest.json');reports=objects(report)
    if rank==0:
        result=dict(passed=all(r['passed'] for r in reports),world_size=8,rounds=8,global_decisions=sum(r['decisions'] for r in reports),
            global_ticks=sum(r['ticks'] for r in reports),global_terminals=sum(r['terminals'] for r in reports),joint_actor_steps=sum(s['metrics']['optimizer_steps'] for s in summaries),
            joint_critic_steps=sum(s['metrics']['critic_optimizer_steps'] for s in summaries),seconds=max(r['seconds'] for r in reports),final_actor=policy.digest(),rank_reports=reports,
            alpha=1.,r2_enabled=False,state2_enabled=False,stimulation=False,actual_joint_synchronization=True)
        b.write(out/'report.json',result);c.manifest(out,'campaign_manifest.json');b.emit('eightgpu_collection_finished',passed=result['passed'],global_decisions=result['global_decisions'],seconds=result['seconds'])
    global_assert(all(r['passed'] for r in reports),'Resource screen failed');dist.barrier()

def verify(out,rank):
    if rank==0:c.check_manifest(out,'campaign_manifest.json')
    dist.barrier();own=out/f'rank_{rank:02d}';report=b.read(own/'report.json');assert report['passed'] and report['collection_pid']!=os.getpid()
    assert b.read(own/'identity.json')==source_identity();uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()[rank].strip()
    assert uuid==report['gpu']['uuid'];policy=b.new_policy();results=[];seed=PROTOCOL['world_seed']+rank*10000;previous=b.CONFIG['warm_checkpoint'];last=None
    with c.environment() as env:
        obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
        for number in range(1,9):
            rd=own/f'round_{number:02d}';events=[json.loads(s) for s in (rd/'rollout.jsonl').read_text().splitlines()];rows=[e['training'] for e in events];summary=b.read(rd/'report.json');boot=b.read(rd/'bootstrap.json')
            assert len(rows)==16;assert all(summary['checks'].values())
            b.load(policy,previous);assert c.state(policy)==summary['before']
            if last:assert summary['before']==last['after'] and summary['optimizer_steps_before']==last['optimizer_steps_after'] and summary['optimizer_hash_before']==last['optimizer_hash_after']
            for i,(e,r) in enumerate(zip(events,rows)):
                assert e['index']==i and e['seed']==seed and e['observation']==obs
                assert e['prepared']['branches'][r['verb']][r['target']]==e['action']
                nxt=env.step(e['action']);score=tracker.score(nxt['transition'],nxt['observation'])
                assert nxt['transition']==e['transition'] and nxt['observation']==e['next_observation'] and nxt['terminated']==r['terminal']
                assert score==e['r1_score'] and r['reward']==score['reward'] and r['duration']==score['elapsed_ticks']
                assert e['reward_record']==dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
                obs=nxt['observation']
                if r['terminal']:
                    seed+=1;obs=env.reset(seed)['observation'];assert e['reset']==dict(seed=seed,observation=obs);tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
                else:assert 'reset' not in e
            c.audit_gae(rows,boot['value'],b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
            pp=policy.prepare([e['observation'] for e in events]);assert [b.plain(p) for p in pp]==[e['prepared'] for e in events]
            for r,p in zip(rows,pp):r['prepared']=p
            old=b.max_difference(c.statistics(policy,rows),b.row_statistics(rows));assert old<=1e-6
            with torch.no_grad():boot2=c.finish(policy,rows,obs)
            assert boot2['prepared']==boot['prepared'] and boot2['terminal']==boot['terminal'];be=abs(boot2['value']-boot['value']);assert be<=1e-6
            checkpoint=out/f'checkpoint_{number:02d}';b.load(policy,checkpoint);new=b.max_difference(c.statistics(policy,rows),b.read(rd/'post_update_statistics.json'));assert new<=1e-6
            assert c.state(policy)==summary['after'];oh,steps=saved_optimizer_hash(checkpoint/'optimizers.pt');assert oh==summary['optimizer_hash_after'] and steps==summary['optimizer_steps_after']
            optimizer_gate(summary['optimizer_steps_before'],steps,summary['metrics']['optimizer_steps'],summary['metrics']['critic_optimizer_steps'])
            result=dict(round=number,rank=rank,decisions=16,old_error=old,new_error=new,bootstrap_error=be,native_replay=True,gae_verified=True,optimizer_verified=True)
            results.append(result);previous=checkpoint;last=summary;del rows,events,pp;c.resources(policy)
            all_results=objects(result)
            if rank==0:b.emit('eightgpu_independent_round_passed',round=number,max_old_error=max(r['old_error'] for r in all_results),max_new_error=max(r['new_error'] for r in all_results),max_bootstrap_error=max(r['bootstrap_error'] for r in all_results))
    assert source_identity()==b.read(own/'identity.json')
    result=dict(passed=True,rank=rank,uuid=uuid,collection_pid=report['collection_pid'],verification_pid=os.getpid(),rounds=results)
    b.write(own/'independent_verification.json',result);all_results=objects(result)
    if rank==0:b.write(out/'independent_verification.json',dict(passed=True,ranks=all_results,campaign_manifest_sha256=b.sha(out/'campaign_manifest.json')));b.emit('eightgpu_independent_passed',ranks=8,decisions=1024)
    dist.barrier()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','verify']);p.add_argument('output',type=Path);args=p.parse_args();rank=init()
    try:(collect if args.action=='collect' else verify)(args.output,rank)
    finally:dist.destroy_process_group()
