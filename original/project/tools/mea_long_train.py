"""Fixed long-recording offline training and independent replay; no device API."""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import observation_key
from agent.mea_protocol import canonical,digest,integer,number
from agent.mea_h5_replay import H5MappedRecording
from agent.mea_reward import record_reward
from agent.mea_snapshot import FEATURE_NAMES
from agent.rl_contract import gae
from tools.mea_offline import (CONFIG as SHORT_CONFIG,evidence,policy_statistics,
    validate_policy_statistics,_tensor_hash,_value,source_identity as base_identity)
from tools.mea_p0 import ROOT,DEFAULT_GODOT,sha,read_json,write_json,_same

CONFIG=SHORT_CONFIG|dict(minibatch=128,critic_minibatch=128)
PROTOCOL=dict(schema='mea.long.offline.v2',chip_id='28080',mapping_basis='finalized_h5_metadata',
    hardware_mapping_independently_verified=False,rounds=8,decisions_per_round=128,seed=5100101,
    train_current=[0,1024],bootstrap_index=1024,gap=[1025,1085],heldout=[1085,1686],
    alpha=1.,r2_enabled=False,realtime_training=False,stimulation=False,
    minimum_windows=1686,association='sequential_offline',biological_effect_verified=False)
CHECK_NAMES={'actor_changed','critic_changed','frozen_base_unchanged','reference_unchanged',
    'saved_inputs_unchanged','new_critic_columns_learned','frozen_base_has_no_gradient',
    'parameters_finite','optimizer_identity_unchanged'}


def emit(event,**fields):print(canonical(dict(event=event,**fields)),flush=True)


def window_split(count):
    integer(count,'window count',PROTOCOL['minimum_windows'])
    return {k:PROTOCOL[k] for k in ('train_current','bootstrap_index','gap','heldout')}


def finish_trajectory(rows,bootstrap,gamma,lam):
    if not rows or not 0<gamma<1 or not 0<lam<=1:raise ValueError('invalid GAE arguments')
    number(bootstrap,'bootstrap')
    for i,r in enumerate(rows):
        if type(r['terminal']) is not bool:raise ValueError('terminal must be boolean')
        integer(r['duration'],'duration',1);number(r['reward'],'reward');number(r['value'],'value')
        r['boundary']=r['terminal'] or i==len(rows)-1
        r['next_value']=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else bootstrap
    advantages,returns=gae([r['reward'] for r in rows],[r['value'] for r in rows],
        [r['next_value'] for r in rows],[r['duration'] for r in rows],[r['terminal'] for r in rows],
        [r['boundary'] for r in rows],gamma,lam)
    for r,a,v in zip(rows,advantages,returns):r.update(advantage=a,**{'return':v})


def audit_gae(rows,bootstrap,gamma,lam):
    carry=0.
    for i in reversed(range(len(rows))):
        r=rows[i];n=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else bootstrap
        boundary=r['terminal'] or i==len(rows)-1
        _same(r['boundary'],boundary,'GAE boundary');_same(r['next_value'],n,'GAE next value')
        d=gamma**r['duration'];carry=r['reward']+d*n-r['value']+d*lam**r['duration']*(not boundary)*carry
        if abs(number(r['advantage'],'advantage')-carry)>1e-10 or abs(number(r['return'],'return')-(carry+r['value']))>1e-10:
            raise ValueError('GAE audit differs')


def optimizer_gate(before,after,actor_steps,critic_steps):
    for name,increment in [('actor',actor_steps),('critic',critic_steps)]:
        integer(increment,'optimizer increment',1)
        old,new=before[name],after[name]
        if not new or (old and len(old)!=len(new)):raise ValueError('optimizer state count changed')
        if old and any(s!=old[0] for s in old):raise ValueError('mixed old optimizer steps')
        expected=(old[0] if old else 0)+increment
        if any(s!=expected for s in new):raise ValueError('optimizer reset or inconsistent increment')


def validate_initial_checkpoint(original,loaded,manifest,stored,registration_sha):
    _same(loaded,original,'initial checkpoint must equal reconstructed original policy')
    for k,v in dict(update=0,registration_sha256=registration_sha,actor_sha256=original['actor']).items():
        _same(manifest.get(k),v,'initial checkpoint '+k)
    for k in ('actor','critic'):
        _same(stored[k]['state'],{},'initial optimizer must be empty: '+k)


def validate_round_evidence(sr,before,after,initial,rows,old_error,new_columns_learned,finite_parameters,ablation=None):
    names=CHECK_NAMES if ablation is None else CHECK_NAMES-{'new_critic_columns_learned'}|{'neural_columns_as_expected'}
    if set(sr['checks'])!=names or any(v is not True for v in sr['checks'].values()):raise ValueError('round check schema')
    derived=dict(actor_changed=before['actor']!=after['actor'],critic_changed=before['critic']!=after['critic'],
        frozen_base_unchanged=after['base']==before['base']==initial['base'],
        reference_unchanged=after['reference']==before['reference']==initial['reference'],
        new_critic_columns_learned=new_columns_learned,parameters_finite=finite_parameters)
    if any(v is not True for v in derived.values()):raise ValueError('checkpoint evidence rejects update')
    for k,v in dict(before=before,after=after,decisions=128,ticks=sum(r['duration'] for r in rows),
            terminal_indices=[r['index'] for r in rows if r['terminal']],
            maximum_input_tokens=max(len(r['prepared']['input_ids']) for r in rows)).items():
        _same(sr[k],v,'replayed round '+k)
    actor=sr['applied_steps']['actor'];integer(actor,'actor steps',1)
    if ablation is not None:
        from agent.mea_long_ablation import validate_budget
        validate_budget(sr['applied_steps'])
    if actor>CONFIG['epochs']:raise ValueError('actor update budget exceeded')
    _same(sr['applied_steps']['critic'],CONFIG['critic_epochs'],'critic update budget')
    _same(sr['metrics']['optimizer_steps'],actor,'metrics/optimizer hooks')
    for k in ('pre_update_statistic_error','roundtrip_error'):
        if not 0<=number(sr[k],k)<=1e-6:raise ValueError('numerical tolerance violated')
    if abs(sr['pre_update_statistic_error']-old_error)>1e-6:raise ValueError('reported old statistic error differs')


def steps(opts):
    return {name:[int(s['step'].item()) for s in opt.state.values() if 'step' in s] for name,opt in opts.items()}


def state(policy):
    return dict(actor=policy.digest(),critic=_tensor_hash(policy.critic.named_parameters()),
        base=_tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad),
        reference=_tensor_hash(policy.reference_model.named_parameters()))


def identity(project,godot,model_config,ablation=None):
    result=dict(sources=base_identity(project)|{'code/tools/mea_long_train.py':sha(Path(__file__))},
        engine=sha(godot),config=CONFIG,protocol=PROTOCOL,model_paths=read_json(model_config))
    if ablation is not None:result['ablation']=ablation
    return result


def new_policy(model_config,ablation=None):
    import torch
    from agent.mea_policy import RecordingPolicy
    paths=read_json(model_config)
    if set(paths)!=set(('model','initial_checkpoint','parent_checkpoint','warm_checkpoint')):raise ValueError('model path keys')
    if ablation is not None:
        from agent.mea_ablation_policy import AblationPolicy
        from agent.mea_long_ablation import cell_config
        _same(ablation,cell_config(ablation['block'],ablation['arm']),'registered ablation config')
        torch.manual_seed(ablation['initial_seed'])
        return AblationPolicy(SimpleNamespace(**CONFIG,**paths,rng_seed=ablation['initial_seed']),ablation['arm'])
    torch.manual_seed(PROTOCOL['seed'])
    return RecordingPolicy(SimpleNamespace(**CONFIG,**paths,rng_seed=PROTOCOL['seed']))


def prepare(policy,data,index,obs):
    snapshot=data.snapshot(index,observation_key(obs))
    return snapshot,policy.prepare([obs],[None if getattr(policy,'arm','real')=='none' else snapshot])[0]


def collect_rollout(policy,data,env,obs,tracker,seed,start,count,partial=None,action_seed=None):
    rows=[]
    for i in range(start,start+count):
        snapshot,prepared=prepare(policy,data,i,obs)
        if action_seed is not None:
            import torch
            torch.manual_seed(action_seed+i)
        sampled=policy.act([prepared],with_reference=True);action=sampled['actions'][0]
        nxt=env.step(action);score=tracker.score(nxt['transition'],nxt['observation'])
        ns=None if nxt['terminated'] else data.snapshot(i+1,observation_key(nxt['observation']))
        reward=record_reward(score['reward'],neural_metrics=dict(zip(FEATURE_NAMES,snapshot.feature_values)))
        row=dict(index=i,seed=seed,observation=obs,snapshot=snapshot.to_dict(),prepared=prepared,
            action=action,action_source='model',transition=nxt['transition'],next_observation=nxt['observation'],
            next_snapshot=ns.to_dict() if ns else None,terminal=nxt['terminated'],r1_score=score,
            reward_record=reward,reward=reward['total_reward'],duration=score['elapsed_ticks'],value=sampled['values'][0])
        for key,plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                           ('reference_verb','reference_verb'),('reference_target','reference_target')]:row[key]=sampled[plural][0]
        obs=nxt['observation']
        if row['terminal']:
            seed+=1;obs=env.reset(seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
            row['reset']=dict(seed=seed,observation=obs)
        rows.append(row)
        if partial:
            with partial.open('a',encoding='utf-8') as f:f.write(canonical(row|{'prepared':evidence(prepared)})+'\n')
        if (i+1)%16==0:emit('collection_progress',index=i,tick=obs['tick'],seed=seed)
    if rows[-1]['terminal']:bootstrap=dict(terminal=True,value=0.,snapshot=None,prepared=None)
    else:
        bs,bp=prepare(policy,data,start+count,obs)
        bootstrap=dict(terminal=False,value=_value(policy,bp),snapshot=bs.to_dict(),prepared=evidence(bp))
    finish_trajectory(rows,bootstrap['value'],CONFIG['gamma_tick'],CONFIG['gae_lambda'])
    audit_gae(rows,bootstrap['value'],CONFIG['gamma_tick'],CONFIG['gae_lambda'])
    return rows,bootstrap,obs,tracker,seed


def stats(policy,rows):return policy_statistics(policy,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])


def seal(out,name):
    write_json(out/name,{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})


def check_manifest(out,name,extra=()):
    expected=read_json(out/name)
    files={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}-{name,*extra}
    if files!=set(expected):raise ValueError('run inventory differs')
    for n,h in expected.items():
        p=(out/n).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p)!=h:raise ValueError('run file differs: '+n)


def recording(source,pin,ablation=None):
    data=H5MappedRecording(source,manifest_sha256=pin,expected_chip_id=PROTOCOL['chip_id'])
    if ablation is not None:
        from agent.mea_long_ablation import AssignedRecording
        data=AssignedRecording(data,ablation)
    return data


def column_check(policy,ablation):
    import torch
    actual=bool(torch.count_nonzero(policy.critic[0].weight[:,108:]))
    if ablation is None:return actual
    from agent.mea_long_ablation import columns_expected
    return columns_expected(ablation['arm'],actual)


def train(out,source,pin,model_config,project,godot,ablation=None):
    import torch
    from agent.m8_train import update
    from agent.m9_policy import save,load
    data=recording(source,pin,ablation);split=window_split(data.window_count)
    out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    registration=dict(identity=identity(project,godot,model_config,ablation),source=data.summary(),split=split,collection_pid=os.getpid())
    write_json(out/'registration.json',registration);emit('loading_model',source_windows=data.window_count)
    policy=new_policy(model_config,ablation);initial=state(policy);write_json(out/'initial_parameters.json',initial)
    if any('lora_' not in n for n,p in policy.model.named_parameters() if p.requires_grad):raise ValueError('non-LoRA trainable')
    opts=dict(actor=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=CONFIG['actor_lr'],eps=1e-5,weight_decay=0),
        critic=torch.optim.AdamW(policy.critic.parameters(),lr=CONFIG['critic_lr'],eps=1e-5,weight_decay=0))
    param_ids={k:[id(p) for g in o.param_groups for p in g['params']] for k,o in opts.items()}
    counts=dict(actor=0,critic=0)
    def hook(k):
        def notify(opt,args,kwargs):counts[k]+=1
        return notify
    hooks=[opt.register_step_post_hook(hook(k)) for k,opt in opts.items()]
    save(policy,out/'initial_checkpoint',opts['actor'],opts['critic'],0,digest(registration))
    reports=[]
    with GodotEnv(str(godot),str(project),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        seed=PROTOCOL['seed'] if ablation is None else ablation['world_seed'];obs=env.reset(seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        for number_round in range(1,PROTOCOL['rounds']+1):
            rd=out/f'round_{number_round:02d}';rd.mkdir();started=time.monotonic()
            before=state(policy);before_steps=steps(opts);before_counts=counts.copy()
            rows,boot,obs,tracker,seed=collect_rollout(policy,data,env,obs,tracker,seed,
                (number_round-1)*128,128,rd/'collection.partial.jsonl',None if ablation is None else ablation['action_seed'])
            with (rd/'rollout.jsonl').open('x',encoding='utf-8') as f:
                for r in rows:f.write(canonical(r|{'prepared':evidence(r['prepared'])})+'\n')
            write_json(rd/'bootstrap.json',boot)
            input_hash=digest([evidence(r['prepared']) for r in rows])
            pre_error=validate_policy_statistics(stats(policy,rows),rows)
            emit('update_started',round=number_round,all_old_statistics_error=pre_error)
            if ablation is not None:torch.manual_seed(ablation['ppo_seed']+number_round)
            metrics=update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**CONFIG))
            after=state(policy);after_steps=steps(opts);applied={k:counts[k]-before_counts[k] for k in counts}
            _same(applied['actor'],metrics['optimizer_steps'],'actor step hooks');_same(applied['critic'],16,'critic step hooks')
            optimizer_gate(before_steps,after_steps,applied['actor'],applied['critic'])
            if ablation is not None:
                from agent.mea_long_ablation import validate_budget
                validate_budget(applied)
            checks=dict(actor_changed=after['actor']!=before['actor'],critic_changed=after['critic']!=before['critic'],
                frozen_base_unchanged=after['base']==initial['base'],reference_unchanged=after['reference']==initial['reference'],
                saved_inputs_unchanged=digest([evidence(r['prepared']) for r in rows])==input_hash,
                new_critic_columns_learned=bool(torch.count_nonzero(policy.critic[0].weight[:,108:])),
                frozen_base_has_no_gradient=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad),
                parameters_finite=all(bool(torch.isfinite(p).all()) for mod in (policy.model,policy.critic) for p in mod.parameters()),
                optimizer_identity_unchanged=param_ids=={k:[id(p) for g in o.param_groups for p in g['params']] for k,o in opts.items()})
            if ablation is not None:
                del checks['new_critic_columns_learned']
                checks['neural_columns_as_expected']=column_check(policy,ablation)
            if not all(checks.values()):raise ValueError('update checks failed: '+canonical(checks))
            post=stats(policy,rows);write_json(rd/'post_update_statistics.json',post)
            save(policy,rd/'checkpoint',opts['actor'],opts['critic'],number_round,digest(registration))
            load(policy,rd/'checkpoint');roundtrip=validate_policy_statistics(stats(policy,rows),post)
            _same(steps(opts),after_steps,'optimizer continuity across checkpoint roundtrip')
            summary=dict(round=number_round,decisions=len(rows),ticks=sum(r['duration'] for r in rows),
                terminal_indices=[r['index'] for r in rows if r['terminal']],before=before,after=after,checks=checks,
                optimizer_steps_before=before_steps,optimizer_steps_after=after_steps,applied_steps=applied,
                pre_update_statistic_error=pre_error,roundtrip_error=roundtrip,metrics=metrics,
                seconds=time.monotonic()-started,maximum_input_tokens=max(len(r['prepared']['input_ids']) for r in rows))
            write_json(rd/'report.json',summary);reports.append(summary)
            del rows,post;policy.clear_prefix_cache();gc.collect();torch.cuda.synchronize()
            emit('round_passed',round=number_round,seconds=summary['seconds'],applied_steps=applied)
    _same(identity(project,godot,model_config,ablation),registration['identity'],'code/model config identity')
    report=dict(status='trained_pending_independent_verification',source=data.summary(),protocol=PROTOCOL,
        model_training_performed=True,rounds=reports,decisions=1024,elapsed_ticks=sum(r['ticks'] for r in reports),
        terminals=sum(len(r['terminal_indices']) for r in reports),seconds=time.monotonic()-began,
        initial=initial,final=state(policy),stimulation=False,device_commands_sent=0,alpha=1.,r2_enabled=False,
        r2_valid=False,realtime_training=False,heldout_used=False,biological_effect_verified=False)
    write_json(out/'report.json',report);seal(out,'training_manifest.json');emit('training_finished',decisions=1024,seconds=report['seconds'])


def verify(out,source,pin,model_config,project,godot,ablation=None):
    import torch
    from agent.m9_policy import load,verify_checkpoint
    check_manifest(out,'training_manifest.json')
    registration=read_json(out/'registration.json');report=read_json(out/'report.json')
    _same(registration['identity'],identity(project,godot,model_config,ablation),'source/config identity')
    if registration['collection_pid']==os.getpid():raise ValueError('fresh verifier process required')
    data=recording(source,pin,ablation);_same(data.summary(),registration['source'],'source pin')
    _same(window_split(data.window_count),registration['split'],'data split')
    policy=new_policy(model_config,ablation);original=state(policy)
    _same(original,read_json(out/'initial_parameters.json'),'original tensor identity')
    previous=out/'initial_checkpoint';results=[];total_ticks=0;terminals=0
    load(policy,previous)
    initial_stored=torch.load(previous/'optimizers.pt',map_location='cpu',weights_only=False)
    validate_initial_checkpoint(original,state(policy),verify_checkpoint(previous),initial_stored,digest(registration))
    expected_steps=dict(actor=[],critic=[]);previous_state=original
    del initial_stored
    with GodotEnv(str(godot),str(project),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        seed=PROTOCOL['seed'] if ablation is None else ablation['world_seed'];obs=env.reset(seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        for number_round in range(1,9):
            rd=out/f'round_{number_round:02d}';saved=[read_json_line(s) for s in (rd/'rollout.jsonl').read_text().splitlines()]
            sr=read_json(rd/'report.json');boot=read_json(rd/'bootstrap.json')
            if len(saved)!=128:raise ValueError('wrong decision count')
            load(policy,previous);before=state(policy)
            _same(before,previous_state,'previous checkpoint continuity')
            _same(before,sr['before'],'old policy identity')
            _same(sr['optimizer_steps_before'],expected_steps,'previous optimizer baseline');rows=[]
            for offset,row in enumerate(saved):
                i=(number_round-1)*128+offset
                _same(row['index'],i,'sequential index');_same(row['seed'],seed,'episode seed');_same(row['observation'],obs,'native observation')
                s,p=prepare(policy,data,i,obs);_same(s.to_dict(),row['snapshot'],'recorded window');_same(evidence(p),row['prepared'],'tokens and inputs')
                _same(p['branches'][row['verb']][row['target']],row['action'],'model action')
                _same(row['action_source'],'model','model-only collection')
                nxt=env.step(row['action']);score=tracker.score(nxt['transition'],nxt['observation'])
                for key,actual in [('transition',nxt['transition']),('next_observation',nxt['observation']),('terminal',nxt['terminated']),('r1_score',score),('reward',score['reward']),('duration',score['elapsed_ticks'])]:_same(row[key],actual,key)
                _same(row['reward_record'],record_reward(score['reward'],neural_metrics=dict(zip(FEATURE_NAMES,s.feature_values))),'R1-only neural telemetry')
                ns=None if nxt['terminated'] else data.snapshot(i+1,observation_key(nxt['observation']))
                _same(row['next_snapshot'],ns.to_dict() if ns else None,'next snapshot')
                obs=nxt['observation'];total_ticks+=score['elapsed_ticks'];terminals+=int(nxt['terminated'])
                if nxt['terminated']:
                    seed+=1;obs=env.reset(seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
                    _same(row.get('reset'),dict(seed=seed,observation=obs),'terminal reset')
                elif 'reset' in row:raise ValueError('nonterminal reset')
                rows.append(row|{'prepared':p})
            _same(boot['terminal'],rows[-1]['terminal'],'bootstrap terminal')
            if rows[-1]['terminal']:_same(boot,dict(terminal=True,value=0.,snapshot=None,prepared=None),'terminal bootstrap');boot_error=0.
            else:
                bs,bp=prepare(policy,data,number_round*128,obs)
                _same(bs.to_dict(),boot['snapshot'],'boundary snapshot');_same(evidence(bp),boot['prepared'],'boundary inputs')
                boot_error=abs(_value(policy,bp)-boot['value'])
                if boot_error>1e-6:raise ValueError('boundary value differs')
            audit_gae(rows,boot['value'],CONFIG['gamma_tick'],CONFIG['gae_lambda'])
            old_error=validate_policy_statistics(stats(policy,rows),rows)
            load(policy,rd/'checkpoint');after=state(policy);_same(after,sr['after'],'new tensor identity')
            new_error=validate_policy_statistics(stats(policy,rows),read_json(rd/'post_update_statistics.json'))
            validate_round_evidence(sr,before,after,original,rows,old_error,
                column_check(policy,ablation),
                all(bool(torch.isfinite(p).all()) for mod in (policy.model,policy.critic) for p in mod.parameters()),ablation)
            cm=verify_checkpoint(rd/'checkpoint');_same(cm['registration_sha256'],digest(registration),'checkpoint registration')
            _same(cm['update'],number_round,'checkpoint round')
            stored=torch.load(rd/'checkpoint/optimizers.pt',map_location='cpu',weights_only=False)
            stored_steps={k:[int(s['step'].item()) for s in stored[k]['state'].values() if 'step' in s] for k in ('actor','critic')}
            _same(stored_steps,sr['optimizer_steps_after'],'serialized optimizer steps')
            optimizer_gate(sr['optimizer_steps_before'],stored_steps,sr['applied_steps']['actor'],sr['applied_steps']['critic'])
            expected_steps=stored_steps;previous_state=after
            _same(sr,report['rounds'][number_round-1],'round summary')
            result=dict(round=number_round,old_error=old_error,new_error=new_error,bootstrap_error=boot_error,
                native_replay=True,recorded_inputs=True,gae_verified=True,optimizer_steps_after=stored_steps)
            results.append(result);previous=rd/'checkpoint';emit('verified_round',**{k:v for k,v in result.items() if k!='optimizer_steps_after'})
            del rows,saved,stored;policy.clear_prefix_cache();gc.collect()
    expected=dict(status='trained_pending_independent_verification',source=data.summary(),protocol=PROTOCOL,
        model_training_performed=True,decisions=1024,elapsed_ticks=total_ticks,terminals=terminals,
        initial=read_json(out/'initial_parameters.json'),final=state(policy),stimulation=False,device_commands_sent=0,
        alpha=1.,r2_enabled=False,r2_valid=False,realtime_training=False,heldout_used=False,biological_effect_verified=False)
    for k,v in expected.items():_same(report[k],v,'completion '+k)
    _same(identity(project,godot,model_config,ablation),registration['identity'],'source stable')
    result=dict(passed=True,status='accepted_offline_training',verification_pid=os.getpid(),
        collection_pid=registration['collection_pid'],training_manifest_sha256=sha(out/'training_manifest.json'),
        source_manifest_sha256=pin,decisions=1024,rounds=results,heldout_used=False,biological_effect_verified=False,
        training_time_only_assertions=['frozen_base_has_no_gradient','optimizer_identity_unchanged'],
        verification_limits=['Adam moment evolution and random sampling stream are not numerically reproduced.'])
    write_json(out/'independent_verification.json',result);emit('independent_verification_passed',decisions=1024)


def read_json_line(line):
    from agent.mea_protocol import strict_json
    return strict_json(line)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','verify']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--model-config',type=Path,required=True);p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--godot',type=Path,default=DEFAULT_GODOT)
    a=p.parse_args()
    if len(a.manifest_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.manifest_sha256):p.error('explicit lowercase SHA256 required')
    try:(train if a.action=='train' else verify)(a.output,a.source,a.manifest_sha256,a.model_config,a.project,a.godot)
    except Exception as exc:
        emit('failed',action=a.action,error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
