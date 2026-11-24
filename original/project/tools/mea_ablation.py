"""Registered recording-only three-arm experiment. No device imports or commands."""
import argparse
import math
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_ablation import ARMS,protocol,assignment,encode_arm,episode_metrics,analyze
from agent.mea_contract import observation_key
from agent.mea_replay import AuditedRecording,RecordingSnapshot,MANIFEST_SHA256
from agent.mea_reward import record_reward
from agent.mea_snapshot import FEATURE_NAMES
from agent.mea_protocol import canonical,digest,strict_json
from tools.mea_offline import (CONFIG,ROOT,DEFAULT_GODOT,sha,read_json,write_json,evidence,
    finish_rows,_value,_tensor_hash,_logps,policy_statistics,validate_policy_statistics,source_identity,_same,_action)


def identities(project):
    return source_identity(project)|{'code/tools/'+name:sha(Path(__file__).with_name(name))
        for name in ('mea_ablation.py','mea_ablation_batch.py')}


def tensor_identity(policy):
    return dict(actor=policy.digest(),critic=_tensor_hash(policy.critic.named_parameters()),
        base=_tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad),
        reference=_tensor_hash(policy.reference_model.named_parameters()))


def validate_common_start(actual,registered):
    if set(actual)!={'actor','critic','base','reference'}:raise ValueError('common start fields')
    _same(actual,registered,'registered common tensors')


def validate_update_claim(update):
    keys={'actor_changed','critic_changed','base_unchanged','reference_unchanged',
          'inputs_unchanged','base_no_gradient','neural_columns_as_expected'}
    if set(update['checks'])!=keys or any(v is not True for v in update['checks'].values()):
        raise ValueError('missing or failed update checks')
    error=update['saved_statistic_error']
    if not isinstance(error,(float,int)) or not math.isfinite(error) or not 0<=error<=1e-6:
        raise ValueError('saved statistic error')
    if type(update['metrics']['optimizer_steps']) is not int or update['metrics']['optimizer_steps']!=2:
        raise ValueError('actor update budget')


def validate_loaded_change(before,after,columns,arm,steps):
    if (after['actor']==before['actor'] or after['critic']==before['critic'] or
        after['base']!=before['base'] or after['reference']!=before['reference'] or
        columns!=(arm!='none') or steps!={'actor':[2],'critic':[16]}):
        raise ValueError('independent checkpoint did not satisfy actual update invariants')


def validate_receipt(receipt,block,arm,registration_sha,manifest_sha):
    for k,v in dict(passed=True,model_verified=True,block=block,arm=arm,native_episodes=5,
        registration_sha256=registration_sha,manifest_sha256=manifest_sha).items():_same(receipt.get(k),v,'receipt '+k)
    errors=receipt.get('statistic_errors',{})
    if set(errors)!={'pre0','pre1','train','post0','post1'} or any(
        not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1e-6 for v in errors.values()):
        raise ValueError('receipt probability/value checks')


def prepare(policy,obs,snapshot,arm):
    visible=None if arm=='none' else snapshot
    return policy.prepare([obs],[visible])[0] if policy else encode_arm(obs,visible,arm)


def episode(policy,data,arm,order,world_seed,action_seed,output,godot,project):
    if arm not in ARMS or len(order)<2 or len(order)!=len(set(order)) or any(not 0<=i<data.window_count for i in order):
        raise ValueError('invalid finite window assignment')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[]
    with GodotEnv(str(godot),str(project),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        obs=env.reset(world_seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        s=data.snapshot(order[0],observation_key(obs));p=prepare(policy,obs,s,arm)
        for i in range(len(order)-1):
            if policy:
                import torch
                torch.manual_seed(action_seed+i)
                sampled=policy.act([p],with_reference=True);action=sampled['actions'][0]
            else:sampled=None;action=_action(p,i)
            nxt=env.step(action);nobs=nxt['observation'];terminal=nxt['terminated'];score=tracker.score(nxt['transition'],nobs)
            ns=None if terminal else data.snapshot(order[i+1],observation_key(nobs))
            np=None if terminal else prepare(policy,nobs,ns,arm)
            reward=record_reward(score['reward'],neural_metrics=dict(zip(FEATURE_NAMES,s.feature_values)))
            row=dict(index=i,arm=arm,action_source='model' if policy else 'scripted_test',observation=obs,
                snapshot=s.to_dict(),prepared=p,action=action,transition=nxt['transition'],next_observation=nobs,
                next_snapshot=ns.to_dict() if ns else None,terminal=terminal,r1_score=score,reward_record=reward,
                reward=reward['total_reward'],duration=score['elapsed_ticks'],value=sampled['values'][0] if policy else 0.)
            if policy:
                for key,plural in [('verb','verbs'),('target','targets'),('logp','logp'),('verb_logp','verb_logp'),
                    ('reference_verb','reference_verb'),('reference_target','reference_target')]:row[key]=sampled[plural][0]
            rows.append(row)
            with (out/'collection.partial.jsonl').open('a',encoding='utf-8') as f:f.write(canonical(row|{'prepared':evidence(p)})+'\n')
            if terminal:break
            obs,s,p=nobs,ns,np
        boot=dict(terminal=rows[-1]['terminal'],value=0. if rows[-1]['terminal'] else _value(policy,p),
            snapshot=None if rows[-1]['terminal'] else s.to_dict(),prepared=None if rows[-1]['terminal'] else evidence(p))
    finish_rows(rows,boot['value'],CONFIG['gamma_tick'],CONFIG['gae_lambda'])
    with (out/'rollout.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:f.write(canonical(r|{'prepared':evidence(r['prepared'])})+'\n')
    metrics=episode_metrics(rows)
    write_json(out/'bootstrap.json',boot);write_json(out/'metrics.json',metrics)
    write_json(out/'assignment.json',dict(arm=arm,order=order,world_seed=world_seed,action_seed=action_seed))
    return rows,boot,metrics


def replay_episode(rows,boot,metrics,data,arm,order,world_seed,godot,project):
    if not rows or len(rows)>len(order)-1:raise ValueError('episode length')
    with GodotEnv(str(godot),str(project),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        obs=env.reset(world_seed)['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        for i,r in enumerate(rows):
            _same(r['index'],i,'index');_same(r['arm'],arm,'arm');_same(obs,r['observation'],'observation')
            s=data.snapshot(order[i],observation_key(obs));_same(s.to_dict(),r['snapshot'],'snapshot')
            base=encode_arm(obs,None if arm=='none' else s,arm)
            for k,v in base.items():_same(v,r['prepared'][k],'prepared '+k)
            if r['action_source']=='model':_same(base['branches'][r['verb']][r['target']],r['action'],'legal action')
            elif r['action_source']=='scripted_test':_same(_action(base,i),r['action'],'test action')
            else:raise ValueError('action source')
            nxt=env.step(r['action']);score=tracker.score(nxt['transition'],nxt['observation'])
            for k,v in [('transition',nxt['transition']),('next_observation',nxt['observation']),('terminal',nxt['terminated']),
                        ('r1_score',score),('reward',score['reward']),('duration',score['elapsed_ticks'])]:_same(v,r[k],k)
            _same(record_reward(score['reward'],neural_metrics=dict(zip(FEATURE_NAMES,s.feature_values))),r['reward_record'],'R1-only record')
            ns=None if nxt['terminated'] else data.snapshot(order[i+1],observation_key(nxt['observation']))
            _same(ns.to_dict() if ns else None,r['next_snapshot'],'next snapshot')
            if i+1<len(rows):_same(r['next_snapshot'],rows[i+1]['snapshot'],'next reused')
            if r['terminal'] and i!=len(rows)-1:raise ValueError('continued after death')
            obs=nxt['observation']
    _same(boot['terminal'],rows[-1]['terminal'],'bootstrap terminal')
    _same(boot['snapshot'],rows[-1]['next_snapshot'],'bootstrap snapshot')
    if rows[-1]['terminal']:
        _same(boot['value'],0.,'terminal value');_same(boot['prepared'],None,'terminal prepared')
    else:
        if len(rows)!=len(order)-1:raise ValueError('unexplained truncation')
        for k,v in encode_arm(obs,None if arm=='none' else ns,arm).items():_same(v,boot['prepared'][k],'bootstrap '+k)
    expected=deepcopy(rows);finish_rows(expected,boot['value'],CONFIG['gamma_tick'],CONFIG['gae_lambda'])
    for r,e in zip(rows,expected):
        for k in ('boundary','next_value','advantage','return'):_same(r[k],e[k],'GAE '+k)
    _same(episode_metrics(rows),metrics,'metrics')
    return True


def cell_schedule(block,arm):
    p=protocol();train=assignment(arm,'train',block);evaluation=assignment(arm,'eval',block)
    return [('pre'+str(j),evaluation,p['eval_world_seed']+2*block+j,p['eval_action_seed']+100*block+30*j) for j in range(2)]+[
        ('train',train,p['train_world_seed']+block,p['train_action_seed']+100*block)]+[
        ('post'+str(j),evaluation,p['eval_world_seed']+2*block+j,p['eval_action_seed']+100*block+30*j) for j in range(2)]


def register(root,source,godot,project,model_config):
    data=AuditedRecording(source)
    if data.window_count!=57:raise ValueError('registration requires the57-window source')
    policy=make_policy(model_config,0,'none')
    baseline=tensor_identity(policy)
    out=Path(root);out.mkdir(parents=True,exist_ok=False)
    registration=dict(protocol=protocol(),config=CONFIG,source=data.summary(),sources=identities(project),
        engine_sha256=sha(godot),model_config=read_json(model_config),common_initial_tensors=baseline,
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'registration.json',registration)
    return digest(registration)


def registration(root,godot,project):
    r=read_json(Path(root)/'registration.json')
    _same(r['protocol'],protocol(),'registered protocol');_same(r['config'],CONFIG,'config')
    _same(r['sources'],identities(project),'source identity');_same(r['engine_sha256'],sha(godot),'engine')
    return r


def make_policy(model_config,block,arm):
    from agent.mea_ablation_policy import AblationPolicy
    c=read_json(model_config)
    if set(c)!={'model','initial_checkpoint','parent_checkpoint','warm_checkpoint'}:raise ValueError('model path fields')
    return AblationPolicy(SimpleNamespace(**CONFIG,**c,rng_seed=protocol()['initial_seed']+block),arm)


def update_policy(policy,rows,out,baseline,block,registration_sha):
    import torch
    from agent.m8_train import update
    from agent.m9_policy import save
    inputs=digest([evidence(r['prepared']) for r in rows])
    stats=policy_statistics(policy,[r['prepared'] for r in rows],[r['verb'] for r in rows],[r['target'] for r in rows])
    error=validate_policy_statistics(stats,rows)
    if any('lora_' not in n for n,p in policy.model.named_parameters() if p.requires_grad):raise ValueError('unfrozen base')
    a=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=CONFIG['actor_lr'],eps=1e-5,weight_decay=0)
    c=torch.optim.AdamW(policy.critic.parameters(),lr=CONFIG['critic_lr'],eps=1e-5,weight_decay=0)
    torch.manual_seed(protocol()['ppo_seed']+block)
    result=update(policy,a,c,rows,SimpleNamespace(**CONFIG))
    checks=dict(actor_changed=policy.digest()!=baseline['actor'],critic_changed=_tensor_hash(policy.critic.named_parameters())!=baseline['critic'],
        base_unchanged=_tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad)==baseline['base'],
        reference_unchanged=_tensor_hash(policy.reference_model.named_parameters())==baseline['reference'],
        inputs_unchanged=digest([evidence(r['prepared']) for r in rows])==inputs,
        base_no_gradient=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad))
    columns=bool(torch.count_nonzero(policy.critic[0].weight[:,108:]))
    checks['neural_columns_as_expected']=columns==(policy.arm!='none')
    if not all(checks.values()) or result['optimizer_steps']!=2:raise ValueError('update/budget check: '+canonical(checks))
    save(policy,out/'checkpoint',a,c,1,registration_sha)
    return dict(checks=checks,metrics=result,saved_statistic_error=error,actor_sha256=policy.digest(),neural_columns_changed=columns)


def run_cell(root,source,model_config,block,arm,godot,project):
    reg=registration(root,godot,project);schedule=cell_schedule(block,arm);out=Path(root)/f'b{block:02d}_{arm}'
    _same(read_json(model_config),reg['model_config'],'registered model paths')
    out.mkdir(exist_ok=False);data=AuditedRecording(source);_same(data.summary(),reg['source'],'source')
    try:
        policy=make_policy(model_config,block,arm)
        baseline=tensor_identity(policy);validate_common_start(baseline,reg['common_initial_tensors'])
        summaries={};update=None
        for name,order,world_seed,action_seed in schedule:
            rows,boot,metrics=episode(policy,data,arm,order,world_seed,action_seed,out/name,godot,project)
            summaries[name]=metrics
            if name=='train':
                if len(rows)!=32:raise ValueError('incomplete training budget; no replacement world')
                update=update_policy(policy,rows,out,baseline,block,digest(reg))
        _same(identities(project),reg['sources'],'source after run')
        _same(_tensor_hash(policy.reference_model.named_parameters()),baseline['reference'],'reference after evaluation')
        report=dict(status='complete',block=block,arm=arm,pre=[summaries['pre0'],summaries['pre1']],
            post=[summaries['post0'],summaries['post1']],train=summaries['train'],baseline=baseline,update=update,
            alpha=1.,r2_enabled=False,stimulation=False,device_commands_sent=0,realtime_training=False)
        write_json(out/'report.json',report)
        write_json(out/'manifest.json',dict(registration_sha256=digest(reg),
            files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
        return report
    except Exception as e:
        write_json(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)))
        raise


def load_episode(path):
    return ([strict_json(s) for s in (path/'rollout.jsonl').read_text(encoding='utf-8').splitlines()],
        read_json(path/'bootstrap.json'),read_json(path/'metrics.json'))


def verify_cell(root,source,model_config,block,arm,godot,project,model=False):
    reg=registration(root,godot,project);out=Path(root)/f'b{block:02d}_{arm}'
    if (out/'failure.json').exists():raise ValueError('failed cell')
    manifest=read_json(out/'manifest.json');_same(manifest['registration_sha256'],digest(reg),'registered run')
    # Nested checkpoint manifests are part of the run inventory.
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
    _same(sorted(actual),sorted(manifest['files']),'inventory')
    for name,h in manifest['files'].items():
        p=(out/name).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p)!=h:raise ValueError('file changed: '+name)
    report=read_json(out/'report.json');data=AuditedRecording(source);_same(data.summary(),reg['source'],'source')
    metrics={};episodes={}
    for name,order,world_seed,action_seed in cell_schedule(block,arm):
        rows,boot,score=load_episode(out/name)
        _same(read_json(out/name/'assignment.json'),dict(arm=arm,order=order,world_seed=world_seed,action_seed=action_seed),'assignment')
        if any(r['action_source']!='model' for r in rows):raise ValueError('efficacy requires actual model actions')
        replay_episode(rows,boot,score,data,arm,order,world_seed,godot,project);metrics[name]=score;episodes[name]=(rows,boot)
    for k,v in dict(status='complete',block=block,arm=arm,pre=[metrics['pre0'],metrics['pre1']],post=[metrics['post0'],metrics['post1']],
        train=metrics['train'],alpha=1.,r2_enabled=False,stimulation=False,device_commands_sent=0,realtime_training=False).items():_same(report[k],v,'report '+k)
    if report['train']['decisions']!=32:raise ValueError('training budget')
    validate_update_claim(report['update']);validate_common_start(report['baseline'],reg['common_initial_tensors'])
    checkpoint=read_json(out/'checkpoint/manifest.json')
    _same(checkpoint['registration_sha256'],digest(reg),'checkpoint registration')
    _same(checkpoint['actor_sha256'],report['update']['actor_sha256'],'checkpoint actor')
    errors={}
    if model:
        import torch
        from agent.mea_ablation_policy import load_arm
        _same(read_json(model_config),reg['model_config'],'registered model paths')
        policy=make_policy(model_config,block,arm)
        baseline=tensor_identity(policy);validate_common_start(baseline,reg['common_initial_tensors'])
        for name in ('pre0','pre1','train','post0','post1'):
            if name=='post0':load_arm(policy,out/'checkpoint')
            rows,boot=episodes[name]
            prepared=policy.prepare([r['observation'] for r in rows],
                [None if arm=='none' else RecordingSnapshot.from_dict(r['snapshot']) for r in rows])
            for p,r in zip(prepared,rows):_same(evidence(p),r['prepared'],'independent saved input/tokens')
            # All pre/train statistics; four post probes per independent evaluation world.
            n=4 if name.startswith('post') else len(rows)
            stats=policy_statistics(policy,prepared[:n],[r['verb'] for r in rows[:n]],[r['target'] for r in rows[:n]])
            errors[name]=validate_policy_statistics(stats,rows[:n])
            if not boot['terminal']:
                bp=prepare(policy,rows[-1]['next_observation'],RecordingSnapshot.from_dict(boot['snapshot']),arm)
                _same(evidence(bp),boot['prepared'],'independent boundary input')
                if abs(_value(policy,bp)-boot['value'])>1e-6:raise ValueError('boundary value differs')
        policy.assert_reference_frozen()
        _same(policy.digest(),report['update']['actor_sha256'],'reloaded actor')
        _same(errors['train'],report['update']['saved_statistic_error'],'reported update guard')
        states=torch.load(out/'checkpoint/optimizers.pt',map_location='cpu',weights_only=True)
        steps={k:sorted({int(s['step']) for s in states[k]['state'].values()}) for k in ('actor','critic')}
        validate_loaded_change(baseline,tensor_identity(policy),bool(torch.count_nonzero(policy.critic[0].weight[:,108:])),arm,steps)
    return dict(passed=True,block=block,arm=arm,native_episodes=5,model_verified=model,statistic_errors=errors,
        registration_sha256=digest(reg),manifest_sha256=sha(out/'manifest.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','cell','verify','analyze'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--godot',type=Path,default=DEFAULT_GODOT);p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--model-config',type=Path);p.add_argument('--block',type=int);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--model',action='store_true');p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.command=='register':result={'registration_sha256':register(a.root,a.source,a.godot,a.project,a.model_config)}
    elif a.command=='cell':result=run_cell(a.root,a.source,a.model_config,a.block,a.arm,a.godot,a.project)
    elif a.command=='verify':result=verify_cell(a.root,a.source,a.model_config,a.block,a.arm,a.godot,a.project,a.model)
    else:
        reg=registration(a.root,a.godot,a.project)
        reports=[]
        for b in range(8):
            for arm in ARMS:
                path=a.root/f'b{b:02d}_{arm}'
                receipt=a.root/f'verified_b{b:02d}_{arm}.json'
                if path.exists() and (path/'manifest.json').exists() and receipt.exists():
                    v=read_json(receipt)
                    validate_receipt(v,b,arm,digest(reg),sha(path/'manifest.json'))
                    verify_cell(a.root,a.source,None,b,arm,a.godot,a.project)
                    reports.append(read_json(path/'report.json'))
        result=analyze(reports)
    if a.result:write_json(a.result,result)
    print(canonical(result))


if __name__=='__main__':main()
