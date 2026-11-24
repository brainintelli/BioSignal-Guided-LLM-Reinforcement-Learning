"""Bounded state1/R1-only GPU acceptance. No device/MEA acquisition imports."""
import argparse,hashlib,json,math,os,socket,time
from pathlib import Path
from types import SimpleNamespace
import torch
from agent.m9_policy import Policy,save,load
from agent.m9_env import GodotEnv
from agent.m9_contract import Reward
from agent.m9_run import finish_trajectory
from agent.m8_train import update
from tools.benchmark_m8s3 import selected_logps

ROOT=Path('/home/researcher/mea-r2-22554')
WORK=ROOT/'smoke_20261116_01'
ASSETS=ROOT/'import_20261116_01/assets'
RUNTIME=ROOT/'runtime_20261116_01/assets'
PROJECT=WORK/'project'
GODOT=RUNTIME/'godot/Godot_v4.7.2-stable_linux.x86_64'
EXPECTED='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'
CONFIG=dict(model=str(ASSETS/'models/Qwen3.5-0.8B'),
 initial_checkpoint=str(ASSETS/'m8b-runs/exploration_attempt1/checkpoint_0032'),
 parent_checkpoint=str(ASSETS/'m6-runs/small_v2_stable/checkpoint_0016'),
 warm_checkpoint=str(ASSETS/'m9-runs/research_attempt5/confirm/combined_r2/checkpoint_0064'),
 rng_seed=4116101,map_mode='infinite_freeze',navigation_steps=4,state_variant='reserves',reward_variant='bottleneck',
 gamma_tick=.99985,gae_lambda=.999,actor_lr=2e-6,critic_lr=1e-4,epochs=2,critic_epochs=16,
 minibatch=32,microbatch=4,critic_minibatch=32,clip=.1,reference_kl=.02,entropy_coef=.002,
 actor_grad_clip=1.,critic_grad_clip=1.,target_kl=.015,zero_update_logp_tolerance=1e-6,
 max_tokens=4096,verb_exploration_epsilon=.1,alpha=1.,r2_enabled=False,state2_enabled=False)

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):
    with Path(path).open('x') as f:f.write(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def emit(event,**kwargs):print(canonical(dict(event=event,time=time.time(),**kwargs)),flush=True)
def plain(p):return json.loads(canonical({k:v for k,v in p.items() if k!='semantic_trees'}))
def tensor_hash(items):
    h=hashlib.sha256()
    for name,p in items:
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def source_identity():
    m=read(ASSETS/'bundle_manifest.json')
    for name,item in m['files'].items():assert sha(ASSETS/name)==item['sha256'],name
    for name,item in m['files'].items():
        prefix='m9-research-v5-project/'
        if name.startswith(prefix):assert sha(PROJECT/name[len(prefix):])==item['sha256'],name
    return dict(asset_manifest=sha(ASSETS/'bundle_manifest.json'),harness=sha(__file__),godot=sha(GODOT))
def max_difference(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        return max((max_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b),(len(a),len(b))
        return max((max_difference(x,y) for x,y in zip(a,b)),default=0.)
    assert math.isfinite(a) and math.isfinite(b)
    return abs(a-b)
@torch.no_grad()
def statistics(policy,prepared,verbs,targets):
    result=[]
    for start in range(0,len(prepared),4):
        pp=prepared[start:start+4];vs=verbs[start:start+4];ts=targets[start:start+4]
        v=torch.tensor(vs,device='cuda');t=torch.tensor(ts,device='cuda')
        vd,td=policy.distributions(pp,vs);rv,rt=policy.distributions(pp,vs,True);values=policy.values(pp)
        for j in range(len(pp)):
            result.append(dict(logp=float(vd.log_prob(v)[j]+td.log_prob(t)[j]),verb_logp=float(vd.log_prob(v)[j]),
                value=float(values[j]),reference_verb=rv.probs[j].tolist(),reference_target=rt.probs[j].tolist()))
    return result
def row_statistics(rows):return [{k:r[k] for k in ['logp','verb_logp','value','reference_verb','reference_target']} for r in rows]
def new_policy():
    assert socket.gethostname()=='researcher-local-6'
    assert CONFIG['alpha']==1 and not CONFIG['r2_enabled'] and not CONFIG['state2_enabled']
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    policy=Policy(SimpleNamespace(**CONFIG))
    assert policy.digest()==EXPECTED and policy.reference_digest()==EXPECTED
    assert policy.critic[0].in_features==108
    assert all('lora_' in n for n,p in policy.model.named_parameters() if p.requires_grad)
    load(policy,CONFIG['warm_checkpoint'])
    policy.assert_reference_frozen()
    return policy
def initial_probe(policy):
    probe=read(Path(CONFIG['warm_checkpoint'])/'reload_probe.json')
    p=policy.prepare([r['observation'] for r in probe['rows']])
    values=selected_logps(policy,p,[r['verb'] for r in probe['rows']],[r['target'] for r in probe['rows']])
    cross_error=max_difference(values,probe['logp'])
    load(policy,CONFIG['warm_checkpoint'])
    repeated=selected_logps(policy,p,[r['verb'] for r in probe['rows']],[r['target'] for r in probe['rows']])
    local_error=max_difference(values,repeated);assert local_error<=1e-6,local_error
    return local_error,len(values),dict(saved_source=probe['logp'],destination=values,destination_reloaded=repeated,
        cross_host_error=cross_error,cross_host_passed=cross_error<=1e-6,local_reload_error=local_error)
def collect(out):
    out.mkdir(exist_ok=False);start=time.monotonic()
    sources=source_identity();write(out/'config.json',CONFIG);write(out/'source_identity.json',sources)
    emit('loading_original');policy=new_policy();probe_error,probe_count,migration_probe=initial_probe(policy)
    write(out/'migration_probe.json',migration_probe)
    emit('original_reloaded',probe_error=probe_error,probe_count=probe_count,actor=policy.digest())
    baseline=dict(actor=policy.digest(),critic=tensor_hash(policy.critic.named_parameters()),
      base=tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad),
      reference=tensor_hash(policy.reference_model.named_parameters()))
    write(out/'initial_parameters.json',baseline)
    rows=[];events=[]
    with GodotEnv(str(GODOT),str(PROJECT),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        obs=env.reset(CONFIG['rng_seed'])['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        for i in range(32):
            prepared=policy.prepare([obs]);sample=policy.act(prepared,with_reference=True)
            action=sample['actions'][0];nxt=env.step(action);score=tracker.score(nxt['transition'],nxt['observation'])
            reward=dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
            row={k:sample[k+'s' if k in ('verb','target','value') else k][0] for k in ['verb','target','value','logp','verb_logp','reference_verb','reference_target']}
            row.update(prepared=prepared[0],reward=reward['total_reward'],duration=score['elapsed_ticks'],terminal=nxt['terminated'],boundary=nxt['terminated'],next_obs=nxt['observation'])
            rows.append(row);events.append(dict(index=i,observation=obs,prepared=plain(prepared[0]),action=action,transition=nxt['transition'],next_observation=nxt['observation'],r1_score=score,reward_record=reward))
            obs=nxt['observation'];emit('decision',index=i,tick=obs['tick'],terminal=nxt['terminated'])
            if nxt['terminated']:break
        if rows[-1]['terminal']:bp=None;bootstrap=0.
        else:
            bp=policy.prepare([obs])[0]
            with torch.no_grad():bootstrap=float(policy.values([bp])[0])
    finish_trajectory(rows,bootstrap,CONFIG['gamma_tick'],CONFIG['gae_lambda'])
    write(out/'bootstrap.json',dict(value=bootstrap,prepared=plain(bp) if bp else None,terminal=rows[-1]['terminal']))
    for event,row in zip(events,rows):event['training']={k:v for k,v in row.items() if k!='prepared'}
    with (out/'rollout.jsonl').open('x') as f:
        for event in events:f.write(canonical(event)+'\n')
    pp=[r['prepared'] for r in rows];vs=[r['verb'] for r in rows];ts=[r['target'] for r in rows]
    before=digest([plain(p) for p in pp]);stats=statistics(policy,pp,vs,ts)
    pre_error=max_difference(stats,row_statistics(rows));assert pre_error<=1e-6,pre_error
    actor_opt=torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad],lr=CONFIG['actor_lr'],eps=1e-5,weight_decay=0)
    critic_opt=torch.optim.AdamW(policy.critic.parameters(),lr=CONFIG['critic_lr'],eps=1e-5,weight_decay=0)
    emit('update_started',decisions=len(rows),pre_update_statistic_error=pre_error)
    metrics=update(policy,actor_opt,critic_opt,rows,SimpleNamespace(**CONFIG));torch.cuda.synchronize()
    checks=dict(actor_changed=policy.digest()!=baseline['actor'],critic_changed=tensor_hash(policy.critic.named_parameters())!=baseline['critic'],
      frozen_base_unchanged=tensor_hash((n,p) for n,p in policy.model.named_parameters() if not p.requires_grad)==baseline['base'],
      reference_unchanged=tensor_hash(policy.reference_model.named_parameters())==baseline['reference'],
      inputs_unchanged=digest([plain(p) for p in pp])==before,
      finite_parameters=all(bool(torch.isfinite(p).all()) for model in [policy.model,policy.critic] for p in model.parameters()),
      no_frozen_gradients=all(p.grad is None for p in policy.model.parameters() if not p.requires_grad),sources_unchanged=source_identity()==sources)
    assert all(checks.values()) and metrics['optimizer_steps']>=1,(checks,metrics)
    post=statistics(policy,pp,vs,ts);write(out/'post_update_statistics.json',post)
    save(policy,out/'checkpoint',actor_opt,critic_opt,1,digest(CONFIG))
    load(policy,out/'checkpoint');roundtrip=max_difference(statistics(policy,pp,vs,ts),post);assert roundtrip<=1e-6
    report=dict(host=socket.gethostname(),collection_pid=os.getpid(),alpha=1.,r2_enabled=False,state2_enabled=False,
      initial_actor=baseline['actor'],final_actor=policy.digest(),decisions=len(rows),ticks=sum(r['duration'] for r in rows),
      terminal=rows[-1]['terminal'],update_calls=1,metrics=metrics,checks=checks,initial_probe_error=probe_error,initial_probe_count=probe_count,
      pre_update_statistic_error=pre_error,roundtrip_error=roundtrip,cross_host_probe_error=migration_probe['cross_host_error'],
      cross_host_probe_passed=migration_probe['cross_host_passed'],fresh_destination_rollout=True,seconds=time.monotonic()-start,
      gpu=torch.cuda.get_device_name(0),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),biological_effect_verified=False)
    write(out/'report.json',report)
    write(out/'collection_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    emit('collection_passed',**report)
def verify(out):
    manifest=read(out/'collection_manifest.json')
    for name,h in manifest.items():assert sha(out/name)==h,name
    assert read(out/'source_identity.json')==source_identity()
    assert read(out/'config.json')==CONFIG
    report=read(out/'report.json');events=[json.loads(line) for line in (out/'rollout.jsonl').read_text().splitlines()]
    assert 1<=len(events)<=32 and len(events)==report['decisions']
    rows=[e['training'] for e in events];bootstrap=read(out/'bootstrap.json')
    with GodotEnv(str(GODOT),str(PROJECT),map_mode=CONFIG['map_mode'],navigation_steps=CONFIG['navigation_steps']) as env:
        obs=env.reset(CONFIG['rng_seed'])['observation'];tracker=Reward(obs,CONFIG['reward_variant'],CONFIG['gamma_tick'])
        for i,(event,row) in enumerate(zip(events,rows)):
            assert event['index']==i and obs==event['observation']
            assert event['prepared']['branches'][row['verb']][row['target']]==event['action']
            nxt=env.step(event['action']);score=tracker.score(nxt['transition'],nxt['observation'])
            assert nxt['transition']==event['transition'] and nxt['observation']==event['next_observation']
            assert nxt['terminated']==row['terminal'] and score==event['r1_score']
            assert row['duration']==score['elapsed_ticks'] and type(row['duration']) is int and row['duration']>0
            assert row['reward']==score['reward'] and event['reward_record']==dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
            assert row['boundary']==(row['terminal'] or i==len(rows)-1)
            if row['terminal']:assert i==len(rows)-1
            obs=nxt['observation']
    assert len(rows)==32 or rows[-1]['terminal']
    assert bootstrap['terminal']==rows[-1]['terminal']
    if rows[-1]['terminal']:assert bootstrap['value']==0. and bootstrap['prepared'] is None
    carry=0.
    for i in reversed(range(len(rows))):
        r=rows[i];n=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else bootstrap['value']
        discount=CONFIG['gamma_tick']**r['duration']
        delta=r['reward']+discount*(not r['terminal'])*n-r['value']
        carry=delta+discount*CONFIG['gae_lambda']**r['duration']*(not r['boundary'])*carry
        assert max_difference([carry,carry+r['value']],[r['advantage'],r['return']])<1e-12
    emit('native_replay_passed',decisions=len(events));policy=new_policy();initial_probe(policy)
    pp=policy.prepare([e['observation'] for e in events])
    assert [plain(p) for p in pp]==[e['prepared'] for e in events]
    vs=[r['verb'] for r in rows];ts=[r['target'] for r in rows]
    old_error=max_difference(statistics(policy,pp,vs,ts),row_statistics(rows));assert old_error<=1e-6
    if not rows[-1]['terminal']:
        bp=policy.prepare([events[-1]['next_observation']])[0];assert plain(bp)==bootstrap['prepared']
        with torch.no_grad():bootstrap_error=max_difference(float(policy.values([bp])[0]),bootstrap['value'])
    else:bootstrap_error=0.
    assert bootstrap_error<=1e-6
    initial=policy.digest();load(policy,out/'checkpoint');new_error=max_difference(statistics(policy,pp,vs,ts),read(out/'post_update_statistics.json'))
    assert new_error<=1e-6 and policy.digest()==report['final_actor']
    result=dict(passed=True,decisions=len(events),initial_actor=initial,final_actor=policy.digest(),reward_exact_matches=len(events),native_transitions=len(events),
      input_identity_verified=True,gae_verified=True,sources_unchanged=source_identity()==read(out/'source_identity.json'),
      old_statistic_error=old_error,new_statistic_error=new_error,bootstrap_error=bootstrap_error,
      collection_pid=report['collection_pid'],verification_pid=os.getpid(),harness_sha256=sha(__file__),collection_manifest_sha256=sha(out/'collection_manifest.json'))
    assert result['collection_pid']!=result['verification_pid'] and result['sources_unchanged']
    write(out/'independent_reload.json',result);emit('independent_verification_passed',**result)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['collect','verify']);parser.add_argument('output',type=Path)
    args=parser.parse_args();(collect if args.action=='collect' else verify)(args.output)
