"""Fixed recorded-data comparison collector; no hardware APIs or stimulation."""
import argparse,copy,math,os,time
from pathlib import Path
import torch
import sr_common as c
from sr_common import b,read,write,digest,canonical
from sr_protocol import PROTOCOL,ARMS,STATE_ARMS,assignment,reward_record,episode_metrics
from sr_contract import context_id
from sr_rollout import finish,validate_sequence

def schedule(block,arm,pilot=False):
    train=list(range(9)) if pilot else assignment(arm,'train',block)
    ev=list(range(10,15)) if pilot else assignment(arm,'eval',block)
    offset=100000 if pilot else 0
    result=[]
    for phase in ('pre','post'):
        for j in range(2):result.append(dict(name=phase+str(j),order=ev,world_seed=PROTOCOL['eval_world_seed']+1000*block+j+offset,action_seed=PROTOCOL['eval_action_seed']+10000*block+1000*j+offset,training=False))
    result.insert(2,dict(name='train',order=train,world_seed=PROTOCOL['train_world_seed']+1000*block+offset,action_seed=PROTOCOL['train_action_seed']+10000*block+offset,training=True))
    return result

def register(root,pilot):
    root.mkdir(exist_ok=False);d=c.data();src=c.source_identity();p=c.policy('baseline',0);common=c.state(p)
    if common['actor']!='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0':raise ValueError('original actor')
    blocks=[0] if pilot else list(range(8));arms=['both'] if pilot else list(ARMS)
    registration=dict(schema='mea.22554.comparison.registration.v1',pilot=pilot,source_identity=src,common_initial=common,protocol=PROTOCOL,
        configs={f'{block}/{arm}':c.config(arm,block) for block in blocks for arm in arms},
        schedules={f'{block}/{arm}':schedule(block,arm,pilot) for block in blocks for arm in arms},
        arm_orders={str(block):arms[block%len(arms):]+arms[:block%len(arms)] for block in blocks},
        baseline_helper='authenticated originalM9 updater/offload',created_time=time.time(),stimulation=False,biological_validated=False)
    write(root/'registration.json',registration);write(root/'registration_pin.json',dict(sha256=c.sha(root/'registration.json')))
    c.emit('registered',pilot=pilot,blocks=len(blocks),arms=len(arms),registration_sha256=c.sha(root/'registration.json'))

def registration(root):
    r=read(root/'registration.json')
    if c.sha(root/'registration.json')!=read(root/'registration_pin.json')['sha256'] or r['source_identity']!=c.source_identity() or r['protocol']!=PROTOCOL:raise ValueError('registration/source changed')
    return r

def prepared(p,data,obs,index):
    snap=data.snapshot(index,context_id(obs));return p.prepare([obs],[snap])[0],snap

def episode(p,data,arm,plan,out):
    out.mkdir(exist_ok=False);order=plan['order'];training=plan['training'];seed=plan['world_seed'];rows=[];events=[];next_snapshot=None
    with c.environment() as env:
        obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
        for i,index in enumerate(order[:-1]):
            pp,s=prepared(p,data,obs,index);torch.manual_seed(plan['action_seed']+i);sample=p.act([pp],with_reference=True)
            action=sample['actions'][0];nxt=env.step(action);score=tracker.score(nxt['transition'],nxt['observation']);reward=reward_record(score['reward'],s['score'],arm,training=training)
            row={k:sample[k+'s' if k in ('verb','target','value') else k][0] for k in ('verb','target','value','logp','verb_logp','reference_verb','reference_target')}
            row.update(prepared=pp,reward=reward['total_reward'],duration=score['elapsed_ticks'],terminal=nxt['terminated'],boundary=nxt['terminated'])
            e=dict(index=i,window=index,seed=seed,observation=obs,snapshot=s,prepared=b.plain(pp),action=action,transition=nxt['transition'],next_observation=nxt['observation'],
                r1_score=score,reward_record=reward,terminal=nxt['terminated'],reset_seed=None,reset_observation=None)
            rows.append(row);events.append(e);obs=nxt['observation']
            if row['terminal'] and training:
                seed+=1;obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick']);e.update(reset_seed=seed,reset_observation=obs)
            with (out/'collection.partial.jsonl').open('a',encoding='utf-8') as f:f.write(canonical(e|dict(training={k:v for k,v in row.items() if k!='prepared'}))+'\n')
            if (i+1)%32==0:c.emit('collection_progress',phase=plan['name'],decisions=i+1)
            if row['terminal'] and not training:break
        def next_value():
            nonlocal next_snapshot
            pp,next_snapshot=prepared(p,data,obs,order[len(rows)])
            with torch.no_grad():v=float(p.values([pp])[0])
            return v,b.plain(pp)
        boot=finish(rows,next_value);boot['snapshot']=next_snapshot;boot['observation']=None if rows[-1]['terminal'] else obs
    validate_sequence(events,order,training=training)
    old_error=b.max_difference(c.stats(p,rows),b.row_statistics(rows))
    if old_error>1e-6:raise ValueError('old statistics differ '+str(old_error))
    with (out/'rollout.jsonl').open('x',encoding='utf-8') as f:
        for row,e in zip(rows,events):f.write(canonical(e|dict(training={k:v for k,v in row.items() if k!='prepared'}))+'\n')
    metrics=episode_metrics([dict(r1=e['r1_score']['reward'],duration=r['duration'],terminal=r['terminal']) for e,r in zip(events,rows)])
    metrics.update(final_observation=events[-1]['next_observation'],old_statistic_error=old_error,raw_aux_sum=sum(e['snapshot']['score']['u'] for e in events),
        optimized_reward_sum=sum(r['reward'] for r in rows),max_input_tokens=max(len(r['prepared']['input_ids']) for r in rows))
    write(out/'bootstrap.json',boot);write(out/'metrics.json',metrics);write(out/'assignment.json',plan)
    c.emit('episode_collected',phase=plan['name'],decisions=len(rows),old_statistic_error=old_error)
    return rows,metrics

def update(p,rows,opts,out,initial,registration_sha):
    before=c.state(p);before_steps=c.steps(opts);before_hash=c.optimizer_hash(opts);inputs=digest([b.plain(r['prepared']) for r in rows]);torch.manual_seed(PROTOCOL['ppo_seed']+p.c.rng_seed-PROTOCOL['initial_seed'])
    c.emit('ppo_started',decisions=len(rows))
    with b.saved_on_cpu_preserve_layout():metrics=b.update(p,opts['actor'],opts['critic'],rows,p.c)
    torch.cuda.synchronize();after=c.state(p);counts=c.steps(opts);opt_hash=c.optimizer_hash(opts)
    critic_steps=p.c.critic_epochs*math.ceil(len(rows)/p.c.critic_minibatch);actor_steps=metrics['optimizer_steps']
    if not 1<=actor_steps<=p.c.epochs*math.ceil(len(rows)/p.c.minibatch) or any(set(counts[k])!={v} for k,v in [('actor',actor_steps),('critic',critic_steps)]):raise ValueError('actual optimizer budget mismatch')
    columns=bool(torch.count_nonzero(p.critic[0].weight[:,108:]).item())
    checks=dict(actor_changed=before['actor']!=after['actor'],critic_changed=before['critic']!=after['critic'],base_unchanged=after['base']==initial['base'],reference_unchanged=after['reference']==initial['reference'],
        inputs_unchanged=inputs==digest([b.plain(r['prepared']) for r in rows]),neural_columns_expected=columns==(p.arm in STATE_ARMS),
        finite_parameters=all(bool(torch.isfinite(v).all()) for m in (p.model,p.critic) for v in m.parameters()),
        finite_gradients=all(v.grad is None or bool(torch.isfinite(v.grad).all()) for m in (p.model,p.critic) for v in m.parameters()),
        frozen_no_gradients=all(v.grad is None for v in p.model.parameters() if not v.requires_grad))
    p.assert_reference_frozen()
    if not all(checks.values()):raise ValueError('update invariant failed '+str(checks))
    post=c.stats(p,rows);write(out/'post_train_statistics.json',post);b.save(p,out/'checkpoint',opts['actor'],opts['critic'],1,registration_sha);c.load(p,out/'checkpoint')
    error=b.max_difference(c.stats(p,rows),post)
    if error>1e-6 or c.state(p)!=after or c.optimizer_hash(opts)!=opt_hash:raise ValueError('checkpoint roundtrip differs')
    disk_hash,disk_steps=c.disk_optimizer(out/'checkpoint/optimizers.pt')
    if disk_hash!=opt_hash or disk_steps!=counts:raise ValueError('serialized optimizer changed')
    result=dict(before=before,after=after,optimizer_before=before_hash,optimizer_after=opt_hash,steps_before=before_steps,steps_after=counts,
        metrics=metrics,critic_steps=critic_steps,checks=checks,reload_error=error)
    write(out/'update.json',result);c.emit('ppo_passed',actor_steps=actor_steps,critic_steps=critic_steps,reload_error=error);return result

def collect(root,out,arm,block):
    r=registration(root);key=f'{block}/{arm}';plans=r['schedules'][key]
    if r['configs'][key]!=c.config(arm,block) or plans!=schedule(block,arm,r['pilot']):raise ValueError('cell differs from registration')
    out.mkdir(exist_ok=False);started=time.monotonic();data=c.data();p=c.policy(arm,block);initial=c.state(p)
    if initial!=r['common_initial']:raise ValueError('common initial tensors differ')
    gpu=c.device()
    if gpu['physical_gpu']!=block:raise ValueError('paired block moved GPU')
    opts=c.optimizers(p);ident={k:[id(v) for g in o.param_groups for v in g['params']] for k,o in opts.items()};reg_sha=c.sha(root/'registration.json')
    b.save(p,out/'initial_checkpoint',opts['actor'],opts['critic'],0,reg_sha);write(out/'identity.json',dict(registration_sha256=reg_sha,initial=initial,gpu=gpu,config=c.config(arm,block),source=r['source_identity']))
    pre=[];post=[];resources=[];update_result=None;train_metrics=None
    for plan in plans:
        rows,metrics=episode(p,data,arm,plan,out/plan['name'])
        if plan['training']:
            if c.state(p)!=initial:raise ValueError('evaluation changed original parameters')
            update_result=update(p,rows,opts,out,initial,reg_sha);train_metrics=metrics
        else:(pre if plan['name'].startswith('pre') else post).append(metrics)
        if ident!={k:[id(v) for g in o.param_groups for v in g['params']] for k,o in opts.items()}:raise ValueError('optimizer object continuity')
        del rows;resources.append(dict(phase=plan['name'],**c.resource(p)))
    if c.state(p)!=update_result['after'] or c.source_identity()!=r['source_identity']:raise ValueError('evaluation/source changed')
    report=dict(passed=True,pilot=r['pilot'],block=block,arm=arm,collection_pid=os.getpid(),gpu=gpu,pre=pre,post=post,train=train_metrics,update=update_result,
        resources=resources,seconds=time.monotonic()-started,registration_sha256=reg_sha,source_full_qc_passed=False,biological_validated=False,stimulation=False)
    write(out/'report.json',report);c.manifest(out,'collection_manifest.json');c.emit('cell_collected',block=block,arm=arm,seconds=report['seconds'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    x=sub.add_parser('register');x.add_argument('--root',type=Path,required=True);x.add_argument('--pilot',action='store_true')
    x=sub.add_parser('collect');x.add_argument('--root',type=Path,required=True);x.add_argument('--output',type=Path,required=True);x.add_argument('--arm',choices=ARMS,required=True);x.add_argument('--block',type=int,required=True)
    a=parser.parse_args()
    if a.action=='register':register(a.root,a.pilot)
    else:collect(a.root,a.output,a.arm,a.block)
