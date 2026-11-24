"""Fixed-length native batch ending in actual death. No state injection."""
import argparse,copy,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
import common as c
from common import b
from batch_boundary import validate_tail
WORLD=4126101;RNG=4146101

def prefix(env,count):
    obs=env.reset(WORLD)['observation'];hashes=[]
    for _ in range(count):
        nxt=env.step({'verb':'wait'});assert not nxt['terminated'];hashes.append(b.digest(nxt['transition']));obs=nxt['observation']
    return obs,hashes

def start(env,fixture):
    obs,hashes=prefix(env,fixture['prefix_steps']);assert hashes==fixture['prefix_hashes'] and obs==fixture['start_observation']
    torch.manual_seed(RNG)
    return obs,b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick'])

def scout():
    policy=b.new_policy();b.initial_probe(policy);attempts=[];selected=None
    with c.environment() as env:
        for count in [488,484,480]:
            obs,hashes=prefix(env,count);first=obs;torch.manual_seed(RNG);tracker=b.Reward(obs,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick']);events=[]
            for i in range(32):
                r,e,obs=c.sample(policy,env,tracker,obs,i,WORLD);events.append(e)
                if r['terminal']:break
            attempts.append(dict(prefix_steps=count,decisions=len(events),terminal=r['terminal'],actions=[e['action'] for e in events]))
            if r['terminal'] and len(events)>=2:
                selected=dict(world_seed=WORLD,rng_seed=RNG,prefix_steps=count,prefix_hashes=hashes,start_observation=first,
                    batch_length=len(events),expected_actions=[e['action'] for e in events],scout_only=True,scripted_prefix_in_ppo=False,attempts=attempts);break
    b.write(b.WORK/'tail_scout_attempts.json',attempts);assert selected,'No qualifying native fixture'
    b.write(b.WORK/'tail_fixture.json',selected);b.emit('tail_fixture_frozen',prefix_steps=count,batch_length=selected['batch_length'])

def collect(out):
    out.mkdir(exist_ok=False);began=time.monotonic();src=c.identity(['terminal_tail.py','tail_fixture.json']);b.write(out/'identity.json',src)
    fixture=b.read(b.WORK/'tail_fixture.json');n=fixture['batch_length'];policy=b.new_policy();_,_,migration=b.initial_probe(policy)
    initial=c.state(policy);rows=[];events=[]
    with c.environment() as env:
        obs,tracker=start(env,fixture)
        for i in range(n):
            r,e,obs=c.sample(policy,env,tracker,obs,i,WORLD);assert r['terminal']==(i==n-1),'Death did not occur at frozen batch tail'
            assert e['action']==fixture['expected_actions'][i];rows.append(r);events.append(e)
        validate_tail(rows,n)
        # Poison callbacks: actual terminal tail must bypass BOTH model calls.
        def forbidden(_):raise AssertionError('model invoked for terminal bootstrap')
        from batch_boundary import bootstrap
        assert bootstrap(rows,obs,forbidden,forbidden)==(0.,None)
        boot=c.finish(policy,rows,obs);assert boot==dict(value=0.,prepared=None,terminal=True)
        altered=copy.deepcopy(rows);b.finish_trajectory(altered,1e9,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
        assert all(r['advantage']==a['advantage'] and r['return']==a['return'] for r,a in zip(rows,altered))
        reset=env.reset(WORLD+1)['observation'];r2,e2,_=c.sample(policy,env,b.Reward(reset,b.CONFIG['reward_variant'],b.CONFIG['gamma_tick']),reset,0,WORLD+1)
        b.write(out/'reset_probe.json',dict(observation=reset,event=e2,excluded_from_ppo=True))
    c.record(out,rows,events,boot);pre=b.max_difference(c.statistics(policy,rows),b.row_statistics(rows));assert pre<=1e-6
    opts=c.optimizers(policy);input_hash=b.digest([b.plain(r['prepared']) for r in rows])
    with b.saved_on_cpu_preserve_layout():metrics=b.update(policy,opts['actor'],opts['critic'],rows,SimpleNamespace(**b.CONFIG))
    after=c.state(policy);checks=c.checks(policy,initial,initial,after,rows,input_hash);assert all(checks.values())
    post=c.statistics(policy,rows);b.write(out/'post_update_statistics.json',post);b.save(policy,out/'checkpoint',opts['actor'],opts['critic'],1,b.digest(src))
    b.load(policy,out/'checkpoint');error=b.max_difference(c.statistics(policy,rows),post);assert error<=1e-6
    assert c.identity(['terminal_tail.py','tail_fixture.json'])==src
    result=dict(passed=True,collection_pid=os.getpid(),decisions=n,ticks=sum(r['duration'] for r in rows),terminal_index=n-1,bootstrap=boot,checks=checks,initial=initial,final=after,
        metrics=metrics,pre_error=pre,reload_error=error,reset_verified=True,external_bootstrap_invariant=True,terminal_bootstrap_skipped_model=True,
        resources=c.resources(policy),seconds=time.monotonic()-began,r2_enabled=False,state2_enabled=False,stimulation=False,alpha=1.,migration=migration)
    b.write(out/'report.json',result);c.manifest(out,'collection_manifest.json');b.emit('tail_collection_passed',decisions=n,terminal_index=n-1,pre_error=pre,reload_error=error)

def verify(out):
    c.check_manifest(out,'collection_manifest.json');assert b.read(out/'identity.json')==c.identity(['terminal_tail.py','tail_fixture.json'])
    report=b.read(out/'report.json');fixture=b.read(b.WORK/'tail_fixture.json');events=[json.loads(s) for s in (out/'rollout.jsonl').read_text().splitlines()]
    rows=[e['training'] for e in events];validate_tail(rows,fixture['batch_length'])
    with c.environment() as env:
        obs,tracker=start(env,fixture)
        for i,(e,r) in enumerate(zip(events,rows)):
            assert e['index']==i and e['observation']==obs and e['seed']==WORLD
            assert e['prepared']['branches'][r['verb']][r['target']]==e['action']
            nxt=env.step(e['action']);score=tracker.score(nxt['transition'],nxt['observation'])
            assert nxt['transition']==e['transition'] and nxt['observation']==e['next_observation'] and nxt['terminated']==r['terminal']
            assert score==e['r1_score'] and r['reward']==score['reward'] and r['duration']==score['elapsed_ticks']
            assert e['reward_record']==dict(alpha=1.,r1=score['reward'],total_reward=score['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
            obs=nxt['observation']
        rp=b.read(out/'reset_probe.json');reset=env.reset(WORLD+1)['observation'];assert reset==rp['observation']
        nxt=env.step(rp['event']['action']);assert nxt['transition']==rp['event']['transition'] and nxt['observation']==rp['event']['next_observation']
    boot=b.read(out/'bootstrap.json');assert boot==dict(value=0.,prepared=None,terminal=True)
    c.audit_gae(rows,0.,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda']);c.audit_gae(rows,1e9,b.CONFIG['gamma_tick'],b.CONFIG['gae_lambda'])
    policy=b.new_policy();pp=policy.prepare([e['observation'] for e in events]);assert [b.plain(p) for p in pp]==[e['prepared'] for e in events]
    for r,p in zip(rows,pp):r['prepared']=p
    old=b.max_difference(c.statistics(policy,rows),b.row_statistics(rows));assert old<=1e-6
    b.load(policy,out/'checkpoint');new=b.max_difference(c.statistics(policy,rows),b.read(out/'post_update_statistics.json'));assert new<=1e-6
    assert os.getpid()!=report['collection_pid'] and c.state(policy)==report['final']
    result=dict(passed=True,collection_pid=report['collection_pid'],verification_pid=os.getpid(),native_replay=True,reset_replay=True,gae_external_bootstrap_invariant=True,
        decisions=len(rows),terminal_index=len(rows)-1,old_error=old,new_error=new,collection_manifest_sha256=b.sha(out/'collection_manifest.json'))
    b.write(out/'independent_verification.json',result);b.emit('tail_independent_passed',**result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['scout','collect','verify']);p.add_argument('output',type=Path,nargs='?');a=p.parse_args()
    if a.action=='scout':scout()
    else:(collect if a.action=='collect' else verify)(a.output)
