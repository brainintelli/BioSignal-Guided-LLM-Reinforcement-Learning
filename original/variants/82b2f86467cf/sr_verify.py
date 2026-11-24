"""Fresh-process native trajectory and complete saved-statistic verification."""
import argparse,math,os
from pathlib import Path
import torch
import sr_common as c
from sr_common import b,read,write
from sr_run import registration,schedule,prepared
from sr_protocol import PROTOCOL,STATE_ARMS,reward_record,episode_metrics
from sr_rollout import audit,validate_sequence

def verify_episode(p,data,arm,plan,out):
    events=[__import__('json').loads(s) for s in (out/'rollout.jsonl').read_text(encoding='utf-8').splitlines()];rows=[];boot=read(out/'bootstrap.json');metrics=read(out/'metrics.json')
    if read(out/'assignment.json')!=plan:raise ValueError('episode assignment changed')
    validate_sequence(events,plan['order'],training=plan['training']);seed=plan['world_seed']
    with c.environment() as env:
        obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
        for i,e in enumerate(events):
            if e['observation']!=obs or e['seed']!=seed or e['window']!=plan['order'][i]:raise ValueError('native/source continuity')
            pp,s=prepared(p,data,obs,plan['order'][i])
            if s!=e['snapshot'] or b.plain(pp)!=e['prepared']:raise ValueError('saved snapshot/token/input mismatch')
            row=e['training']|dict(prepared=pp)
            if pp['branches'][row['verb']][row['target']]!=e['action']:raise ValueError('action identity')
            nxt=env.step(e['action']);score=tracker.score(nxt['transition'],nxt['observation']);reward=reward_record(score['reward'],s['score'],arm,training=plan['training'])
            if any([nxt['transition']!=e['transition'],nxt['observation']!=e['next_observation'],nxt['terminated']!=row['terminal'],nxt['terminated']!=e['terminal'],
                score!=e['r1_score'],reward!=e['reward_record'],row['reward']!=reward['total_reward'],row['duration']!=score['elapsed_ticks']]):raise ValueError('native/reward identity')
            rows.append(row);obs=nxt['observation']
            if row['terminal'] and plan['training']:
                seed+=1;obs=env.reset(seed)['observation'];tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])
                if e['reset_seed']!=seed or e['reset_observation']!=obs:raise ValueError('reset differs')
            elif e['reset_seed'] is not None or e['reset_observation'] is not None:raise ValueError('unexpected reset')
        if rows[-1]['terminal']:
            if boot!=dict(value=0.,prepared=None,terminal=True,snapshot=None,observation=None):raise ValueError('terminal bootstrap invokes model/data')
            bootstrap_error=0.
        else:
            pp,s=prepared(p,data,obs,plan['order'][len(rows)])
            with torch.no_grad():value=float(p.values([pp])[0])
            bootstrap_error=abs(value-boot['value'])
            if boot['terminal'] or s!=boot['snapshot'] or b.plain(pp)!=boot['prepared'] or obs!=boot['observation'] or bootstrap_error>1e-6:raise ValueError('cutoff bootstrap mismatch')
    audit(rows,boot['value'])
    for i,row in enumerate(rows):
        nv=0. if row['terminal'] else rows[i+1]['value'] if i+1<len(rows) else boot['value']
        if row['next_value']!=nv:raise ValueError('saved nextvalue differs')
    error=b.max_difference(c.stats(p,rows),b.row_statistics(rows))
    if error>1e-6:raise ValueError('saved old/new statistics differ '+str(error))
    expected=episode_metrics([dict(r1=e['r1_score']['reward'],duration=r['duration'],terminal=r['terminal']) for e,r in zip(events,rows)])
    if any(metrics[k]!=v for k,v in expected.items()) or metrics['old_statistic_error']>1e-6:raise ValueError('reported task metric differs')
    if metrics['final_observation']!=events[-1]['next_observation'] or metrics['raw_aux_sum']!=sum(e['snapshot']['score']['u'] for e in events) or metrics['optimized_reward_sum']!=sum(r['reward'] for r in rows):raise ValueError('metric/aux decomposition')
    result=dict(phase=plan['name'],decisions=len(rows),native_replay=True,snapshot_inputs_verified=True,reward_gae_verified=True,statistic_error=error,bootstrap_error=bootstrap_error)
    c.emit('independent_phase_passed',**result);return result,rows,metrics

def verify(root,out):
    r=registration(root);c.check_manifest(out,'collection_manifest.json');report=read(out/'report.json');identity=read(out/'identity.json');arm=report['arm'];block=report['block'];key=f'{block}/{arm}';reg_sha=c.sha(root/'registration.json')
    if not report['passed'] or report['collection_pid']==os.getpid() or report['pilot']!=r['pilot'] or report['registration_sha256']!=reg_sha or identity['registration_sha256']!=reg_sha:raise ValueError('cell/receipt identity')
    if identity['source']!=r['source_identity'] or identity['config']!=r['configs'][key] or identity['initial']!=r['common_initial']:raise ValueError('registered identity mismatch')
    gpu=c.device()
    if gpu['physical_gpu']!=block or gpu['uuid']!=report['gpu']['uuid'] or identity['gpu']!=report['gpu']:raise ValueError('same physicalGPU verification required')
    data=c.data();p=c.policy(arm,block);initial=c.state(p)
    if initial!=r['common_initial']:raise ValueError('reconstructed original differs')
    c.load(p,out/'initial_checkpoint')
    im=read(out/'initial_checkpoint/manifest.json');ih,ist=c.disk_optimizer(out/'initial_checkpoint/optimizers.pt');empty=c.optimizer_hash(c.optimizers(p))
    if c.state(p)!=initial or im['update']!=0 or im['registration_sha256']!=reg_sha or ist!=dict(actor=[],critic=[]) or ih!=empty:raise ValueError('initial anchor/optimizer mismatch')
    update=read(out/'update.json')
    if update!=report['update'] or update['before']!=initial or update['optimizer_before']!=empty or update['steps_before']!=ist or not all(v is True for v in update['checks'].values()):raise ValueError('update initial chain/checks')
    results=[];pre=[];post=[];new_train_error=None
    for plan in r['schedules'][key]:
        if plan['name']=='post0':c.load(p,out/'checkpoint')
        phase,rows,metrics=verify_episode(p,data,arm,plan,out/plan['name']);results.append(phase)
        if plan['training']:
            if report['train']!=metrics:raise ValueError('training metric report differs')
            c.load(p,out/'checkpoint');after=c.state(p);new_train_error=b.max_difference(c.stats(p,rows),read(out/'post_train_statistics.json'))
            if after!=update['after'] or after['actor']==initial['actor'] or after['critic']==initial['critic'] or after['base']!=initial['base'] or after['reference']!=initial['reference'] or new_train_error>1e-6:raise ValueError('actual checkpoint change/frozen/updated statistics')
            if bool(torch.count_nonzero(p.critic[0].weight[:,108:]).item())!=(arm in STATE_ARMS):raise ValueError('actual neural columns')
            if not all(bool(torch.isfinite(v).all()) for m in (p.model,p.critic) for v in m.parameters()):raise ValueError('nonfinite saved parameters')
            disk_hash,counts=c.disk_optimizer(out/'checkpoint/optimizers.pt');cm=read(out/'checkpoint/manifest.json')
            critic_steps=p.c.critic_epochs*math.ceil(len(rows)/p.c.critic_minibatch);actor_steps=update['metrics']['optimizer_steps']
            if not 1<=actor_steps<=p.c.epochs*math.ceil(len(rows)/p.c.minibatch) or any(set(counts[k])!={v} for k,v in [('actor',actor_steps),('critic',critic_steps)]):raise ValueError('actual optimizer step budget')
            if disk_hash!=update['optimizer_after'] or counts!=update['steps_after'] or critic_steps!=update['critic_steps'] or cm['update']!=1 or cm['registration_sha256']!=reg_sha:raise ValueError('saved optimizer/checkpoint identity')
        else:(pre if plan['name'].startswith('pre') else post).append(metrics)
        del rows;c.resource(p)
    if pre!=report['pre'] or post!=report['post'] or c.state(p)!=update['after'] or c.source_identity()!=r['source_identity']:raise ValueError('aggregate/source state mismatch')
    result=dict(passed=True,block=block,arm=arm,pilot=r['pilot'],collection_pid=report['collection_pid'],verification_pid=os.getpid(),gpu=gpu,phases=results,
        updated_train_statistic_error=new_train_error,actual_weight_optimizer_checks=True,registration_sha256=reg_sha,collection_manifest_sha256=c.sha(out/'collection_manifest.json'))
    write(out/'independent_verification.json',result);c.emit('cell_independently_verified',block=block,arm=arm,pilot=r['pilot'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);a=p.parse_args();verify(a.root,a.input)
