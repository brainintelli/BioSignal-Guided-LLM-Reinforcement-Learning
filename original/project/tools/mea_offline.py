"""Bounded recording-only state2 / native game / PPO acceptance and replay."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import encode,observation_key
from agent.mea_protocol import canonical,digest,integer,number,strict_json
from agent.mea_replay import AuditedRecording,RecordingSnapshot,MANIFEST_SHA256,RECORDING_CONTRACT
from agent.mea_reward import record_reward
from agent.mea_snapshot import FEATURE_NAMES
from agent.rl_contract import gae
from tools.mea_p0 import ROOT,DEFAULT_GODOT,sha,write_json,read_json,_action,_source_identity,_same

CONFIG=dict(map_mode='infinite_freeze',navigation_steps=4,state_variant='reserves',reward_variant='bottleneck',
    gamma_tick=.99985,gae_lambda=.999,alpha=1.,actor_lr=2e-6,critic_lr=1e-4,epochs=2,critic_epochs=16,
    minibatch=32,microbatch=4,critic_minibatch=32,clip=.1,reference_kl=.02,entropy_coef=.002,
    actor_grad_clip=1.,critic_grad_clip=1.,target_kl=.015,zero_update_logp_tolerance=1e-6,
    max_tokens=4096,verb_exploration_epsilon=.1)


def source_identity(project):return _source_identity(project)|{'code/tools/mea_offline.py':sha(Path(__file__))}


def evidence(prepared):
    return strict_json(canonical({k:v for k,v in prepared.items() if k!='semantic_trees'}))


def finish_rows(rows,bootstrap,gamma,lam):
    if not rows or not 0<gamma<1 or not 0<lam<=1:raise ValueError('invalid trajectory')
    number(bootstrap,'bootstrap')
    for i,row in enumerate(rows):
        if type(row['terminal']) is not bool or (row['terminal'] and i!=len(rows)-1):raise ValueError('invalid terminal')
        integer(row['duration'],'duration',1);number(row['reward'],'reward');number(row['value'],'value')
        row['boundary']=row['terminal'] or i==len(rows)-1
        row['next_value']=0. if row['terminal'] else rows[i+1]['value'] if i+1<len(rows) else bootstrap
    advantages,returns=gae([r['reward'] for r in rows],[r['value'] for r in rows],[r['next_value'] for r in rows],
        [r['duration'] for r in rows],[r['terminal'] for r in rows],[r['boundary'] for r in rows],gamma,lam)
    for row,a,v in zip(rows,advantages,returns):row.update(advantage=a,**{'return':v})


def _prepare(policy,obs,snapshot):return policy.prepare([obs],[snapshot])[0] if policy else encode(obs,snapshot)


def _value(policy,prepared):
    if policy is None:return 0.
    import torch
    with torch.no_grad():return float(policy.values([prepared])[0])


def _tensor_hash(items):
    import torch
    h=hashlib.sha256()
    for name,p in items:
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _logps(policy,prepared,verbs,targets):
    import torch
    values=[]
    with torch.no_grad():
        for start in range(0,len(prepared),4):
            chunk=prepared[start:start+4];v=verbs[start:start+4];t=targets[start:start+4]
            vd,td=policy.distributions(chunk,v)
            values.extend((vd.log_prob(torch.tensor(v,device=policy.device_name))+td.log_prob(torch.tensor(t,device=policy.device_name))).tolist())
    return values


def policy_statistics(policy,prepared,verbs,targets):
    import torch
    result=[]
    with torch.no_grad():
        for start in range(0,len(prepared),4):
            chunk=prepared[start:start+4];v=verbs[start:start+4];t=targets[start:start+4]
            vd,td=policy.distributions(chunk,v);rv,rt=policy.distributions(chunk,v,True)
            vlp=vd.log_prob(torch.tensor(v,device=policy.device_name))
            lp=vlp+td.log_prob(torch.tensor(t,device=policy.device_name));values=policy.values(chunk)
            for i in range(len(chunk)):
                result.append(dict(logp=float(lp[i]),verb_logp=float(vlp[i]),value=float(values[i]),
                    reference_verb=rv.probs[i].tolist(),reference_target=rt.probs[i].tolist()))
    return result


def validate_policy_statistics(actual,saved):
    if not actual or len(actual)!=len(saved):raise ValueError('policy row count differs')
    def delta(a,b):
        if isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b) or not a:raise ValueError('policy probability shape differs')
            return max(delta(x,y) for x,y in zip(a,b))
        return abs(number(a,'recomputed policy value')-number(b,'saved policy value'))
    error=max(delta(a[k],b[k]) for a,b in zip(actual,saved)
              for k in ('logp','verb_logp','value','reference_verb','reference_target'))
    if error>1e-6:raise ValueError('saved actor/reference/value statistics differ')
    return error


def validate_completion_report(report,rows,bootstrap,requested_steps):
    source=rows[0]['action_source'];is_model=source=='model'
    if source not in ('model','scripted_offline_transport') or any(r['action_source']!=source for r in rows):raise ValueError('mixed action sources')
    expected=dict(status='complete',contract=RECORDING_CONTRACT,stimulation=False,device_commands_sent=0,
        alpha=1.,r2_enabled=False,r2_valid=False,r2_raw=None,realtime_training=False,
        model_training_performed=is_model,action_source=source,decisions=len(rows),requested_decisions=requested_steps,
        snapshots=len(rows)+(not rows[-1]['terminal']),elapsed_ticks=sum(r['duration'] for r in rows),
        terminated=rows[-1]['terminal'],total_reward_equals_r1=all(r['reward']==r['r1_score']['reward'] for r in rows),
        critic_input_dim=112,bootstrap_snapshot_id=None if bootstrap['snapshot'] is None else bootstrap['snapshot']['snapshot_id'],
        maximum_input_tokens=max(len(r['prepared'].get('input_ids',[])) for r in rows),biological_effect_verified=False)
    for key,value in expected.items():_same(report.get(key),value,'completion report '+key)
    if not expected['total_reward_equals_r1']:raise ValueError('R1-only reward violated')
    if not is_model:_same(report.get('update'),None,'scripted update absent')
    else:
        update=report.get('update');checks={'actor_changed','critic_changed','frozen_base_unchanged','reference_unchanged',
            'saved_inputs_unchanged','new_critic_columns_learned','frozen_base_has_no_gradient'}
        if not isinstance(update,dict) or set(update.get('checks',{}))!=checks or any(update['checks'][k] is not True for k in checks):
            raise ValueError('model update checks incomplete')
        if (update.get('sample_count')!=len(rows) or update.get('update_calls')!=1 or
            update.get('initial_actor_sha256')!='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0' or
            update.get('final_actor_sha256')==update['initial_actor_sha256'] or update['metrics']['optimizer_steps']<1 or
            number(update['all_old_logp_error'],'old logp error')>1e-6 or
            number(update['all_saved_policy_statistic_error'],'policy statistic error')>1e-6):
            raise ValueError('model update completion inconsistent')


def _update(policy,rows,output,baseline):
    import torch
    from agent.m8_train import update
    from agent.m9_policy import save
    policy.assert_reference_frozen()
    if any('lora_' not in n for n,p in policy.model.named_parameters() if p.requires_grad):raise ValueError('non-LoRA actor trainable')
    prepared=[r['prepared'] for r in rows];before_inputs=digest([evidence(p) for p in prepared])
    actual=policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows])
    statistic_error=validate_policy_statistics(actual,rows)
    zero_error=max(abs(x['logp']-r['logp']) for x,r in zip(actual,rows))
    actor_opt=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=CONFIG['actor_lr'],eps=1e-5,weight_decay=0)
    critic_opt=torch.optim.AdamW(policy.critic.parameters(),lr=CONFIG['critic_lr'],eps=1e-5,weight_decay=0)
    metrics=update(policy,actor_opt,critic_opt,rows,SimpleNamespace(**CONFIG))
    checks=dict(actor_changed=policy.digest()!=baseline['actor'],
        critic_changed=_tensor_hash(policy.critic.named_parameters())!=baseline['critic'],
        frozen_base_unchanged=_tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad)==baseline['base'],
        reference_unchanged=_tensor_hash(policy.reference_model.named_parameters())==baseline['reference'],
        saved_inputs_unchanged=digest([evidence(p) for p in prepared])==before_inputs,
        new_critic_columns_learned=bool(torch.count_nonzero(policy.critic[0].weight[:,108:])),
        frozen_base_has_no_gradient=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad))
    if not all(checks.values()) or metrics['optimizer_steps']<1:raise ValueError('offline update check failed: '+canonical(checks))
    save(policy,output/'checkpoint',actor_opt,critic_opt,1,digest(CONFIG))
    probes=rows[:4]
    with torch.no_grad():values=policy.values([r['prepared'] for r in probes]).tolist()
    write_json(output/'post_update_probe.json',dict(indices=list(range(len(probes))),
        logps=_logps(policy,[r['prepared'] for r in probes],[r['verb'] for r in probes],[r['target'] for r in probes]),
        values=values,actor_sha256=policy.digest()))
    return dict(checks=checks,metrics=metrics,all_old_logp_error=zero_error,all_saved_policy_statistic_error=statistic_error,initial_actor_sha256=baseline['actor'],
                final_actor_sha256=policy.digest(),sample_count=len(rows),update_calls=1)


def collect(output,source,*,godot=DEFAULT_GODOT,project=ROOT,steps=32,seed=4100101,alpha=1.,
            manifest_sha256=MANIFEST_SHA256,policy=None):
    record_reward(0.,alpha=alpha);integer(steps,'steps',1);integer(seed,'seed')
    if steps>32:raise ValueError('bounded engineering run: maximum32 decisions')
    data=AuditedRecording(source,manifest_sha256=manifest_sha256)
    if data.window_count<steps+1:raise ValueError('insufficient current/next windows; no reuse')
    output,project,godot=Path(output),Path(project).resolve(),Path(godot).resolve()
    if not godot.is_file() or not (project/'project.godot').is_file():raise ValueError('native engine/project missing')
    identities,engine=source_identity(project),sha(godot)
    output.mkdir(parents=True,exist_ok=False);rows=[];bootstrap={};baseline=None
    try:
        if policy:
            import torch
            torch.manual_seed(seed)
            baseline=dict(actor=policy.digest(),critic=_tensor_hash(policy.critic.named_parameters()),
                base=_tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad),
                reference=_tensor_hash(policy.reference_model.named_parameters()))
        with GodotEnv(str(godot),str(project),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
            obs=env.reset(seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
            snapshot=data.snapshot(0,observation_key(obs));prepared=_prepare(policy,obs,snapshot)
            for i in range(steps):
                sampled=policy.act([prepared],with_reference=True) if policy else None
                action=sampled['actions'][0] if policy else _action(prepared,i)
                nxt=env.step(action);next_obs=nxt['observation'];terminal=nxt['terminated']
                score=tracker.score(nxt['transition'],next_obs)
                next_snapshot=None if terminal else data.snapshot(i+1,observation_key(next_obs))
                next_prepared=None if terminal else _prepare(policy,next_obs,next_snapshot)
                reward=record_reward(score['reward'],alpha=alpha,neural_metrics=dict(zip(FEATURE_NAMES,snapshot.feature_values)))
                row=dict(index=i,observation=obs,snapshot=snapshot.to_dict(),prepared=prepared,action=action,
                    action_source='model' if policy else 'scripted_offline_transport',transition=nxt['transition'],
                    next_observation=next_obs,next_snapshot=next_snapshot.to_dict() if next_snapshot else None,
                    terminal=terminal,r1_score=score,reward_record=reward,reward=reward['total_reward'],duration=score['elapsed_ticks'],
                    value=sampled['values'][0] if policy else 0.,boundary=False)
                if policy:
                    for key,plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                                       ('reference_verb','reference_verb'),('reference_target','reference_target')]:row[key]=sampled[plural][0]
                rows.append(row)
                with (output/'collection.partial.jsonl').open('a',encoding='utf-8') as f:f.write(canonical(row|{'prepared':evidence(prepared)})+'\n')
                if terminal:break
                obs,snapshot,prepared=next_obs,next_snapshot,next_prepared
            boundary_value=0. if rows[-1]['terminal'] else _value(policy,prepared)
            bootstrap=dict(terminal=rows[-1]['terminal'],value=boundary_value,
                snapshot=None if rows[-1]['terminal'] else snapshot.to_dict(),
                prepared=None if rows[-1]['terminal'] else evidence(prepared))
        finish_rows(rows,boundary_value,CONFIG['gamma_tick'],CONFIG['gae_lambda'])
        with (output/'rollout.jsonl').open('x',encoding='utf-8',newline='\n') as f:
            for row in rows:f.write(canonical(row|{'prepared':evidence(row['prepared'])})+'\n')
        write_json(output/'bootstrap.json',bootstrap)
        update_report=_update(policy,rows,output,baseline) if policy else None
        _same(source_identity(project),identities,'source changed during offline run')
        report=dict(status='complete',contract=RECORDING_CONTRACT,source=data.summary(),stimulation=False,device_commands_sent=0,
            alpha=1.,r2_enabled=False,r2_valid=False,r2_raw=None,realtime_training=False,
            model_training_performed=policy is not None,action_source=rows[0]['action_source'],
            decisions=len(rows),requested_decisions=steps,snapshots=len(rows)+(not rows[-1]['terminal']),
            elapsed_ticks=sum(r['duration'] for r in rows),terminated=rows[-1]['terminal'],
            total_reward_equals_r1=all(r['reward']==r['r1_score']['reward'] for r in rows),
            critic_input_dim=112,bootstrap_snapshot_id=None if rows[-1]['terminal'] else snapshot.snapshot_id,
            maximum_input_tokens=max((len(r['prepared'].get('input_ids',[])) for r in rows),default=0),
            biological_effect_verified=False,update=update_report)
        write_json(output/'report.json',report)
        write_json(output/'manifest.json',dict(schema='mea.offline.run.v1',config=CONFIG,seed=seed,requested_steps=steps,
            source_manifest_sha256=manifest_sha256,sources=identities,engine_sha256=engine,
            created_utc=datetime.now(timezone.utc).isoformat(),files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))
        return report
    except Exception as exc:
        write_json(output/'failure.json',dict(status='invalid',error_type=type(exc).__name__,error=str(exc),collected_decisions=len(rows)))
        raise


def _load_run(root,project,godot):
    root=Path(root)
    if (root/'failure.json').exists():raise ValueError('failed run')
    m=read_json(root/'manifest.json')
    if m.get('schema')!='mea.offline.run.v1':raise ValueError('run schema')
    _same(m['config'],CONFIG,'configuration');_same(m['sources'],source_identity(project),'source identity')
    if sha(godot)!=m['engine_sha256']:raise ValueError('engine identity')
    for name,h in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('run file changed: '+name)
    actual_files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
    if actual_files!=set(m['files']):raise ValueError('run inventory differs')
    rows=[strict_json(s) for s in (root/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
    report=read_json(root/'report.json')
    if not rows or len(rows)!=report['decisions'] or len(rows)>m['requested_steps']:raise ValueError('decision count')
    bootstrap=read_json(root/'bootstrap.json');validate_completion_report(report,rows,bootstrap,m['requested_steps'])
    if report['model_training_performed']:
        checkpoint=read_json(root/'checkpoint/manifest.json')
        _same(checkpoint['actor_sha256'],report['update']['final_actor_sha256'],'checkpoint/update actor')
        _same(checkpoint['registration_sha256'],digest(CONFIG),'checkpoint update config')
    return m,rows,report,bootstrap


def replay(input_dir,source,*,godot=DEFAULT_GODOT,project=ROOT,manifest_sha256=MANIFEST_SHA256):
    root,project,godot=Path(input_dir),Path(project).resolve(),Path(godot).resolve()
    m,rows,report,bootstrap=_load_run(root,project,godot)
    _same(m['source_manifest_sha256'],manifest_sha256,'authorized source identity')
    data=AuditedRecording(source,manifest_sha256=manifest_sha256);_same(data.summary(),report['source'],'source summary')
    with GodotEnv(str(godot),str(project),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        obs=env.reset(m['seed'])['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        for i,row in enumerate(rows):
            if row['index']!=i:raise ValueError('row index')
            _same(obs,row['observation'],'native current observation')
            s=data.snapshot(i,observation_key(obs));_same(s.to_dict(),row['snapshot'],'recorded snapshot')
            base=encode(obs,s)
            for k,v in base.items():_same(v,row['prepared'][k],'prepared '+k)
            if row['action_source']=='scripted_offline_transport':_same(_action(base,i),row['action'],'scripted legal action')
            elif row['action_source']=='model':_same(base['branches'][row['verb']][row['target']],row['action'],'selected legal model action')
            else:raise ValueError('unknown action source')
            nxt=env.step(row['action']);score=tracker.score(nxt['transition'],nxt['observation'])
            _same(nxt['transition'],row['transition'],'native transition');_same(nxt['observation'],row['next_observation'],'native next observation')
            _same(nxt['terminated'],row['terminal'],'terminal');_same(score,row['r1_score'],'R1 components')
            expected=record_reward(score['reward'],neural_metrics=dict(zip(FEATURE_NAMES,s.feature_values)))
            _same(expected,row['reward_record'],'reward record');_same(score['reward'],row['reward'],'R1 exactly')
            _same(score['elapsed_ticks'],row['duration'],'duration')
            ns=None if nxt['terminated'] else data.snapshot(i+1,observation_key(nxt['observation']))
            _same(ns.to_dict() if ns else None,row['next_snapshot'],'next snapshot')
            if i+1<len(rows):_same(row['next_snapshot'],rows[i+1]['snapshot'],'next snapshot reused once')
            if row['terminal'] and i+1!=len(rows):raise ValueError('continued after terminal')
            obs=nxt['observation']
    _same(bootstrap['terminal'],rows[-1]['terminal'],'bootstrap terminal')
    _same(bootstrap['snapshot'],rows[-1]['next_snapshot'],'saved boundary snapshot')
    if rows[-1]['terminal']:
        _same(bootstrap['value'],0.,'terminal bootstrap');_same(bootstrap['prepared'],None,'no terminal observation')
    else:
        for k,v in encode(obs,ns).items():_same(v,bootstrap['prepared'][k],'bootstrap prepared '+k)
        if len(rows)!=m['requested_steps']:raise ValueError('unexplained truncation')
    expected=deepcopy(rows);finish_rows(expected,bootstrap['value'],CONFIG['gamma_tick'],CONFIG['gae_lambda'])
    for r,e in zip(rows,expected):
        for k in ('next_value','advantage','return','boundary'):_same(r[k],e[k],'GAE '+k)
    return dict(passed=True,decisions=len(rows),reward_exact_matches=len(rows),native_transitions=len(rows),
                snapshots=len(rows)+(not rows[-1]['terminal']),saved_bootstrap_verified=True,alpha=1.,
                source='recording',model_probabilities_verified=False,biological_effect_verified=False)


def make_policy(config_path,seed):
    from agent.mea_policy import RecordingPolicy
    settings=read_json(config_path)
    keys={'model','initial_checkpoint','parent_checkpoint','warm_checkpoint'}
    if set(settings)!=keys:raise ValueError('explicit model path fields required')
    return RecordingPolicy(SimpleNamespace(**CONFIG,**settings,rng_seed=seed))


def verify_model(input_dir,source,model_config,*,godot,project):
    import torch
    from agent.m9_policy import load
    root=Path(input_dir);m,rows,report,bootstrap=_load_run(root,Path(project).resolve(),Path(godot))
    if report['model_training_performed'] is not True:raise ValueError('not a model run')
    replay(root,source,godot=godot,project=project)
    policy=make_policy(model_config,m['seed'])
    prepared=policy.prepare([r['observation'] for r in rows],[RecordingSnapshot.from_dict(r['snapshot']) for r in rows])
    for p,r in zip(prepared,rows):_same(evidence(p),r['prepared'],'independent saved tokens/input')
    actual=policy_statistics(policy,prepared,[r['verb'] for r in rows],[r['target'] for r in rows])
    statistic_error=validate_policy_statistics(actual,rows)
    error=max(abs(a['logp']-r['logp']) for a,r in zip(actual,rows))
    value_error=max(abs(a['value']-r['value']) for a,r in zip(actual,rows))
    if not rows[-1]['terminal']:
        bp=policy.prepare([rows[-1]['next_observation']],[RecordingSnapshot.from_dict(bootstrap['snapshot'])])[0]
        _same(evidence(bp),bootstrap['prepared'],'independent bootstrap input')
        value_error=max(value_error,abs(_value(policy,bp)-bootstrap['value']))
    if error>1e-6 or value_error>1e-6:raise ValueError(f'independent old policy mismatch: {error}, {value_error}')
    load(policy,root/'checkpoint');probe=read_json(root/'post_update_probe.json');n=len(probe['indices'])
    post=_logps(policy,prepared[:n],[r['verb'] for r in rows[:n]],[r['target'] for r in rows[:n]])
    with torch.no_grad():post_values=policy.values(prepared[:n]).tolist()
    post_error=max(abs(a-b) for a,b in zip(post,probe['logps']))
    post_value_error=max(abs(a-b) for a,b in zip(post_values,probe['values']))
    if post_error>1e-6 or post_value_error>1e-6 or policy.digest()!=probe['actor_sha256']:raise ValueError('independent new checkpoint differs')
    policy.assert_reference_frozen()
    return dict(passed=True,old_policy_rows=len(rows),old_logp_error=error,old_value_error=value_error,
        reloaded_logp_error=post_error,reloaded_value_error=post_value_error,actor_sha256=policy.digest(),
        saved_policy_statistic_error=statistic_error,reference_probabilities_verified=True,old_verb_logp_verified=True,
        alpha=1.,realtime_training=False,source='recording')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','replay','verify-model'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--result',type=Path)
    p.add_argument('--godot',type=Path,default=DEFAULT_GODOT);p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--model-config',type=Path);p.add_argument('--steps',type=int,default=32)
    p.add_argument('--seed',type=int,default=4100101);p.add_argument('--alpha',type=float,default=1.)
    args=p.parse_args();record_reward(0.,alpha=args.alpha)
    if args.command=='run':
        if args.output is None:p.error('--output required')
        policy=make_policy(args.model_config,args.seed) if args.model_config else None
        result=collect(args.output,args.source,godot=args.godot,project=args.project,steps=args.steps,seed=args.seed,alpha=args.alpha,policy=policy)
    elif args.command=='replay':result=replay(args.input,args.source,godot=args.godot,project=args.project)
    else:result=verify_model(args.input,args.source,args.model_config,godot=args.godot,project=args.project)
    if args.result:write_json(args.result,result)
    print(canonical(result))


if __name__=='__main__':main()
