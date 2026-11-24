"""Offline R1-only campaign on .16; original simulator, no MEA/device API."""
import argparse,gc,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
import base_smoke as b
from protocol import audit_gae,optimizer_gate,resource_gate

b.WORK=Path(__file__).resolve().parent;b.PROJECT=b.WORK/'project'
PROTOCOL=dict(version=1,terminal_seed=4126101,stability_seed=4136101,
    terminal_decisions=32,rounds=8,decisions_per_round=128,
    alpha=1.,r2_enabled=False,state2_enabled=False,stimulation=False,
    source_checkpoint=b.CONFIG['warm_checkpoint'],local_tolerance=1e-6,
    gpu_growth_limit_bytes=512*1024**2,rss_growth_limit_bytes=4*1024**3)

def identity():
    return dict(base=b.source_identity(),driver=b.sha(__file__),protocol_module=b.sha(b.WORK/'protocol.py'),
        fixture=b.sha(b.WORK/'terminal_fixture.json'),protocol=PROTOCOL,config=b.CONFIG)

def state(policy):
    return dict(actor=policy.digest(),critic=b.tensor_hash(policy.critic.named_parameters()),
        base=b.tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad),
        reference=b.tensor_hash(policy.reference_model.named_parameters()))

def steps(opts):
    return {name:[int(s['step'].item()) for s in opt.state.values() if 'step' in s] for name,opt in opts.items()}

def resources(policy):
    policy.clear_prefix_cache();gc.collect();torch.cuda.synchronize()
    rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),rss=rss)

def environment():
    return b.GodotEnv(str(b.GODOT),str(b.PROJECT),map_mode=b.CONFIG['map_mode'],navigation_steps=b.CONFIG['navigation_steps'])

def starting(env,phase):
    seed=PROTOCOL[phase+'_seed'];obs=env.reset(seed)['observation']
    if phase=='terminal':
        f=b.read(b.WORK/'terminal_fixture.json');assert seed==f['seed']
        assert f['source']=='native_legal_wait_prefix' and not f['scripted_actions_in_ppo']
        for h in f['prefix_transition_hashes']:
            nxt=env.step({'verb':'wait'});assert not nxt['terminated'] and b.digest(nxt['transition'])==h
            obs=nxt['observation']
        assert len(f['prefix_transition_hashes'])==f['prefix_steps'] and obs==f['last_alive_observation']
    return seed,obs,b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])

def write_events(path,events):
    with path.open('x') as f:
        for e in events:f.write(b.canonical(e)+'\n')

def manifest(out,name):
    b.write(out/name,{p.relative_to(out).as_posix():b.sha(p) for p in out.rglob('*') if p.is_file()})

def collect(out,phase):
    out.mkdir(exist_ok=False);began=time.monotonic();sources=identity();b.write(out/'identity.json',sources)
    b.emit('loading_original',phase=phase);policy=b.new_policy()
    _,_,migration=b.initial_probe(policy);b.write(out/'migration_probe.json',migration)
    initial=state(policy);b.write(out/'initial_parameters.json',initial)
    opts=dict(actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=b.CONFIG['actor_lr'],eps=1e-5,weight_decay=0),
        critic=torch.optim.AdamW(policy.critic.parameters(),lr=b.CONFIG['critic_lr'],eps=1e-5,weight_decay=0))
    param_ids={name:[id(p) for group in opt.param_groups for p in group['params']] for name,opt in opts.items()}
    hook_counts=dict(actor=0,critic=0)
    def hook(name):
        def notify(opt,args,kwargs):
            hook_counts[name]+=1
            if name=='actor' or hook_counts[name]%16==0:b.emit('optimizer_step',kind=name,count=hook_counts[name])
        return notify
    hooks=[opt.register_step_post_hook(hook(name)) for name,opt in opts.items()]
    rounds=1 if phase=='terminal' else PROTOCOL['rounds'];n=32 if phase=='terminal' else PROTOCOL['decisions_per_round']
    summaries=[];total_terminals=0
    with environment() as env:
        seed,obs,tracker=starting(env,phase)
        for round_no in range(1,rounds+1):
            rd=out/f'round_{round_no:02d}';rd.mkdir();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            before=state(policy);before_steps=steps(opts);before_counts=hook_counts.copy();rows=[];events=[]
            for i in range(n):
                prepared=policy.prepare([obs]);sample=policy.act(prepared,with_reference=True)
                action=sample['actions'][0];nxt=env.step(action);score=tracker.score(nxt['transition'],nxt['observation'])
                row={k:sample[k+'s' if k in ('verb','target','value') else k][0] for k in ['verb','target','value','logp','verb_logp','reference_verb','reference_target']}
                row.update(prepared=prepared[0],reward=score['reward'],duration=score['elapsed_ticks'],terminal=nxt['terminated'],boundary=nxt['terminated'])
                event=dict(index=i,seed=seed,observation=obs,prepared=b.plain(prepared[0]),action=action,transition=nxt['transition'],
                    next_observation=nxt['observation'],r1_score=score,reward_record=dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None))
                rows.append(row);events.append(event);obs=nxt['observation']
                if nxt['terminated']:
                    total_terminals+=1;seed+=1;obs=env.reset(seed)['observation']
                    tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
                    event['reset']=dict(seed=seed,observation=obs)
                    b.emit('native_terminal',round=round_no,index=i,death_tick=nxt['observation']['tick'],reset_seed=seed)
                if (i+1)%16==0:b.emit('collection_progress',round=round_no,decisions=i+1,tick=obs['tick'])
            if rows[-1]['terminal']:bp=None;bootstrap=0.
            else:
                bp=policy.prepare([obs])[0]
                with torch.no_grad():bootstrap=float(policy.values([bp])[0])
            b.finish_trajectory(rows,bootstrap,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
            gae=audit_gae(rows,bootstrap,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
            if phase=='terminal':assert any(i<n-1 for i in gae['terminal_indices']),'No actual terminal followed by new episode'
            b.write(rd/'bootstrap.json',dict(value=bootstrap,prepared=b.plain(bp) if bp is not None else None,terminal=rows[-1]['terminal']))
            for e,r in zip(events,rows):e['training']={k:v for k,v in r.items() if k!='prepared'}
            write_events(rd/'rollout.jsonl',events)
            pp=[r['prepared'] for r in rows];vs=[r['verb'] for r in rows];ts=[r['target'] for r in rows]
            input_hash=b.digest([b.plain(p) for p in pp]);pre_error=b.max_difference(b.statistics(policy,pp,vs,ts),b.row_statistics(rows));assert pre_error<=1e-6
            b.emit('update_started',round=round_no,decisions=n,pre_error=pre_error)
            with b.saved_on_cpu_preserve_layout():metrics=b.update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**b.CONFIG))
            torch.cuda.synchronize();after=state(policy);after_steps=steps(opts)
            applied={k:hook_counts[k]-before_counts[k] for k in hook_counts}
            assert applied['actor']==metrics['optimizer_steps'] and applied['critic']==b.CONFIG['critic_epochs']*((n+31)//32)
            optimizer_gate(before_steps,after_steps,applied['actor'],applied['critic'])
            checks=dict(actor_changed=after['actor']!=before['actor'],critic_changed=after['critic']!=before['critic'],
                frozen_base_unchanged=after['base']==initial['base'],reference_unchanged=after['reference']==initial['reference'],
                inputs_unchanged=b.digest([b.plain(p) for p in pp])==input_hash,
                finite_parameters=all(bool(torch.isfinite(p).all()) for m in [policy.model,policy.critic] for p in m.parameters()),
                finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for m in [policy.model,policy.critic] for p in m.parameters()),
                no_frozen_gradients=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad),
                optimizer_parameter_identity=param_ids=={name:[id(p) for g in opt.param_groups for p in g['params']] for name,opt in opts.items()})
            assert all(checks.values()),checks;policy.assert_reference_frozen()
            post=b.statistics(policy,pp,vs,ts);b.write(rd/'post_update_statistics.json',post)
            b.save(policy,rd/'checkpoint',opts['actor'],opts['critic'],round_no,b.digest(sources))
            b.load(policy,rd/'checkpoint');roundtrip=b.max_difference(b.statistics(policy,pp,vs,ts),post);assert roundtrip<=1e-6
            assert steps(opts)==after_steps and param_ids=={name:[id(p) for g in opt.param_groups for p in g['params']] for name,opt in opts.items()}
            summary=dict(round=round_no,decisions=n,ticks=sum(r['duration'] for r in rows),terminal_indices=gae['terminal_indices'],
                pre_update_statistic_error=pre_error,roundtrip_error=roundtrip,before=before,after=after,checks=checks,
                optimizer_steps_before=before_steps,optimizer_steps_after=after_steps,applied_steps=applied,metrics=metrics)
            del pp,vs,ts,post,prepared,sample,row,event,events,rows,bp
            summary['resources']=resources(policy);summary['seconds']=time.monotonic()-start
            b.write(rd/'report.json',summary);manifest(rd,'collection_manifest.json');summaries.append(summary)
            b.emit('round_passed',round=round_no,seconds=summary['seconds'],resources=summary['resources'],applied_steps=applied)
    assert identity()==sources
    memory=resource_gate([r['resources'] for r in summaries]) if phase=='stability' else None
    report=dict(passed=memory is None or memory['passed'],phase=phase,collection_pid=os.getpid(),rounds=rounds,decisions=rounds*n,
        terminals=total_terminals,seconds=time.monotonic()-began,final_actor=policy.digest(),resource_gate=memory,
        stimulation=False,r2_enabled=False,state2_enabled=False,alpha=1.,fresh_destination_rollout=True,
        cross_host_probe_error=migration['cross_host_error'],cross_host_probe_passed=migration['cross_host_passed'])
    b.write(out/'report.json',report);manifest(out,'campaign_manifest.json');b.emit('campaign_finished',**report)
    assert report['passed'],'Stability resource screen failed'

def verify(out,phase):
    for name,h in b.read(out/'campaign_manifest.json').items():assert b.sha(out/name)==h,name
    assert b.read(out/'identity.json')==identity();report=b.read(out/'report.json');assert report['passed']
    assert report['collection_pid']!=os.getpid();policy=b.new_policy();previous=b.CONFIG['warm_checkpoint'];results=[]
    with environment() as env:
        seed,obs,tracker=starting(env,phase)
        for number in range(1,report['rounds']+1):
            rd=out/f'round_{number:02d}';events=[json.loads(s) for s in (rd/'rollout.jsonl').read_text().splitlines()]
            rows=[e['training'] for e in events];sr=b.read(rd/'report.json');boot=b.read(rd/'bootstrap.json')
            assert len(events)==sr['decisions'];b.load(policy,previous);assert policy.digest()==sr['before']['actor']
            for i,(e,r) in enumerate(zip(events,rows)):
                assert e['index']==i and e['seed']==seed and e['observation']==obs
                assert e['prepared']['branches'][r['verb']][r['target']]==e['action']
                nxt=env.step(e['action']);score=tracker.score(nxt['transition'],nxt['observation'])
                assert nxt['transition']==e['transition'] and nxt['observation']==e['next_observation']
                assert nxt['terminated']==r['terminal'] and score==e['r1_score']
                assert r['reward']==score['reward'] and r['duration']==score['elapsed_ticks']
                assert e['reward_record']==dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
                obs=nxt['observation']
                if r['terminal']:
                    seed+=1;obs=env.reset(seed)['observation'];assert e['reset']==dict(seed=seed,observation=obs)
                    tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
                else:assert 'reset' not in e
            gae=audit_gae(rows,boot['value'],b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
            assert boot['terminal']==rows[-1]['terminal']
            pp=policy.prepare([e['observation'] for e in events]);assert [b.plain(p) for p in pp]==[e['prepared'] for e in events]
            vs=[r['verb'] for r in rows];ts=[r['target'] for r in rows]
            old_error=b.max_difference(b.statistics(policy,pp,vs,ts),b.row_statistics(rows));assert old_error<=1e-6
            if rows[-1]['terminal']:assert boot['value']==0. and boot['prepared'] is None;boot_error=0.
            else:
                bp=policy.prepare([obs])[0];assert b.plain(bp)==boot['prepared']
                with torch.no_grad():boot_error=abs(float(policy.values([bp])[0])-boot['value'])
                assert boot_error<=1e-6
            if phase=='terminal':
                terminal=gae['terminal_indices'][0];assert terminal<len(rows)-1
                import copy
                changed=copy.deepcopy(rows);changed[terminal+1]['reward']+=1e6
                b.finish_trajectory(changed,boot['value'],b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
                assert all(changed[i]['advantage']==rows[i]['advantage'] and changed[i]['return']==rows[i]['return'] for i in range(terminal+1))
            b.load(policy,rd/'checkpoint');new_error=b.max_difference(b.statistics(policy,pp,vs,ts),b.read(rd/'post_update_statistics.json'))
            assert new_error<=1e-6 and policy.digest()==sr['after']['actor']
            stored=torch.load(rd/'checkpoint/optimizers.pt',map_location='cpu',weights_only=False)
            stored_steps={name:[int(s['step'].item()) for s in stored[name]['state'].values() if 'step' in s] for name in ['actor','critic']}
            assert stored_steps==sr['optimizer_steps_after']
            optimizer_gate(sr['optimizer_steps_before'],stored_steps,sr['applied_steps']['actor'],sr['applied_steps']['critic'])
            if results:assert sr['optimizer_steps_before']==results[-1]['optimizer_steps_after']
            result=dict(round=number,decisions=len(rows),terminal_indices=gae['terminal_indices'],old_error=old_error,new_error=new_error,
                bootstrap_error=boot_error,native_replay=True,gae_exact=True,input_identity=True,optimizer_steps_after=stored_steps)
            results.append(result);previous=rd/'checkpoint';b.emit('independent_round_passed',round=number,old_error=old_error,new_error=new_error,bootstrap_error=boot_error)
            del pp,stored,events,rows;resources(policy)
    assert identity()==b.read(out/'identity.json')
    result=dict(passed=True,phase=phase,verification_pid=os.getpid(),collection_pid=report['collection_pid'],rounds=results,
        campaign_manifest_sha256=b.sha(out/'campaign_manifest.json'),source_identity_verified=True)
    b.write(out/'independent_verification.json',result);b.emit('independent_verification_passed',phase=phase,rounds=len(results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['collect','verify']);parser.add_argument('phase',choices=['terminal','stability']);parser.add_argument('output',type=Path)
    args=parser.parse_args();(collect if args.action=='collect' else verify)(args.output,args.phase)
