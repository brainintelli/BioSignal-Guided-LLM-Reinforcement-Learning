"""Local independent evidence/GAE/continuity verification; no GPU dependency."""
import hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence';EXPECTED='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def check_manifest(root,name):
    for f,h in read(root/name).items():assert (root/f).resolve().is_relative_to(root.resolve()) and sha(root/f)==h,f
def gae(rows,boot):
    carry=0.
    for i in reversed(range(len(rows))):
        r=rows[i];assert type(r['duration']) is int and r['duration']>0
        assert r['boundary']==(r['terminal'] or i==len(rows)-1)
        nxt=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else boot
        discount=.99985**r['duration'];carry=r['reward']+discount*nxt-r['value']+discount*.999**r['duration']*(not r['boundary'])*carry
        assert math.isfinite(carry) and abs(carry-r['advantage'])<1e-10 and abs(carry+r['value']-r['return'])<1e-10
def events(rd):
    ee=[json.loads(s) for s in (rd/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
    for i,e in enumerate(ee):
        r=e['training'];assert e['index']==i
        assert e['prepared']['branches'][r['verb']][r['target']]==e['action']
        assert e['r1_score']['reward']==r['reward'] and e['r1_score']['elapsed_ticks']==r['duration']
        assert e['reward_record']==dict(alpha=1.,r1=r['reward'],total_reward=r['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
    return ee
def main():
    m=read(E/'evidence_manifest.json')
    for name,v in m.items():
        p=(E/name).resolve();assert p.is_relative_to(E.resolve()) and p.stat().st_size==v['size'] and sha(p)==v['sha256'],name
    for name in ['common.py','batch_boundary.py','terminal_tail.py','distributed_update.py','eightgpu_run.py','host_cache.py','test_host_cache.py','test_boundary.py','test_distributed.py','test_optimizer_identity.py','process_identity.py','test_process_identity.py','monitor_gpu.py','package_evidence.py']:
        assert sha(R/name)==sha(E/name),name
    old=R.parent/'mea_22554_gpu_smoke_20261116'
    assert sha(E/'base_smoke.py')==sha(old/'checkpoint_smoke.py') and sha(E/'layout_offload.py')==sha(old/'layout_offload.py')
    runtime=read(E/'runtime_audit.json');assert runtime['passed'] and runtime['files']==24551 and runtime['bytes']==5823470108
    for n in ['tail01','tail01_verify','distributed_math','joint03','joint03_verify','monitor_gpu04']:assert (E/(n+'.exit')).read_text().strip()=='0',n
    assert (E/'tail_scout.exit').read_text().strip()!='0' and (E/'joint01.exit').read_text().strip()!='0','failed attempts missing'
    assert (E/'joint02.exit').read_text().strip()!='0' and not read(E/'joint02/report.json')['passed']
    for attempt in ['joint01','joint02']:
        for rank in range(8):
            identity=read(E/attempt/f'rank_{rank:02d}/identity.json')
            assert identity['files']['eightgpu_run.py']==sha(E/f'eightgpu_run_{attempt}.py')
    cleanup=read(E/'post_run_inventory.json');assert cleanup['passed'] and not cleanup['owned_pids'] and not cleanup['compute'].strip()
    tail=E/'tail01';check_manifest(tail,'collection_manifest.json');tr=read(tail/'report.json');tv=read(tail/'independent_verification.json');ee=events(tail);rr=[e['training'] for e in ee]
    for name,h in read(tail/'identity.json')['files'].items():assert sha(E/name)==h
    assert tr['passed'] and tv['passed'] and len(rr)==4 and all(not x['terminal'] for x in rr[:-1]) and rr[-1]['terminal']
    assert tr['collection_pid']!=tv['verification_pid'] and tr['terminal_index']==tv['terminal_index']==3
    assert tr['terminal_bootstrap_skipped_model'] and tr['external_bootstrap_invariant'] and tr['reset_verified'] and all(tr['checks'].values())
    assert read(tail/'bootstrap.json')==dict(value=0.,prepared=None,terminal=True)
    gae(rr,0.);gae(rr,1e9)
    assert max(tr['pre_error'],tr['reload_error'],tv['old_error'],tv['new_error'])<=1e-6
    assert tv['collection_manifest_sha256']==sha(tail/'collection_manifest.json')
    mathcheck=read(E/'distributed_math.json');assert mathcheck['passed'] and len(mathcheck['ranks'])==8 and all(v['passed'] for v in mathcheck['ranks'])
    out=E/'joint03';check_manifest(out,'campaign_manifest.json');root=read(out/'report.json');ind=read(out/'independent_verification.json');inventory=read(out/'device_inventory.json')
    assert root['passed'] and root['world_size']==8 and root['rounds']==8 and root['global_decisions']==1024 and root['actual_joint_synchronization']
    assert root['alpha']==1 and not any(root[k] for k in ['r2_enabled','state2_enabled','stimulation'])
    assert ind['passed'] and ind['campaign_manifest_sha256']==sha(out/'campaign_manifest.json') and len(ind['ranks'])==8
    assert len({d['uuid'] for d in inventory})==8 and len({d['pid'] for d in inventory})==8
    totals=dict(decisions=0,ticks=0,terminals=0);all_rounds=[]
    for rank in range(8):
        own=out/f'rank_{rank:02d}';report=read(own/'report.json');v=read(own/'independent_verification.json');assert v==ind['ranks'][rank]
        assert report['passed'] and report['rank']==rank and v['rank']==rank and v['uuid']==inventory[rank]['uuid']
        assert report['collection_pid']==inventory[rank]['pid']==v['collection_pid']!=v['verification_pid']
        identity=read(own/'identity.json')
        for name,h in identity['files'].items():assert sha(E/name)==h
        last=None;prevobs=None;prevseed=None;rounds=[]
        for number in range(1,9):
            rd=own/f'round_{number:02d}';s=read(rd/'report.json');ivr=v['rounds'][number-1];ee=events(rd);rows=[e['training'] for e in ee];boot=read(rd/'bootstrap.json')
            assert len(rows)==16 and s['decisions']==16 and all(s['checks'].values())
            for x in [s['pre_error'],s['reload_error'],ivr['old_error'],ivr['new_error'],ivr['bootstrap_error']]:assert math.isfinite(x) and x<=1e-6
            assert ivr['native_replay'] and ivr['gae_verified'] and ivr['optimizer_verified'];gae(rows,boot['value'])
            for e,r in zip(ee,rows):
                if prevobs is not None:assert e['observation']==prevobs and e['seed']==prevseed
                if r['terminal']:prevobs=e['reset']['observation'];prevseed=e['reset']['seed'];assert prevseed==e['seed']+1
                else:prevobs=e['next_observation'];prevseed=e['seed']
            if last:assert s['before']==last['after'] and s['optimizer_steps_before']==last['optimizer_steps_after'] and s['optimizer_hash_before']==last['optimizer_hash_after']
            else:assert s['before']['actor']==EXPECTED and s['optimizer_steps_before']==dict(actor=[],critic=[])
            for name,delta in [('actor',s['metrics']['optimizer_steps']),('critic',s['metrics']['critic_optimizer_steps'])]:
                oldstep=max(s['optimizer_steps_before'][name],default=0);assert delta>0 and all(step==oldstep+delta for step in s['optimizer_steps_after'][name])
            assert s['resources']['peak_reserved']>=s['resources']['peak_allocated']>0
            assert s['resources']['peak_rss_process']>=max(s['resources']['rss'],s['resources']['rss_before_host_cache_release'])
            assert all(math.isfinite(value) for value in s['metrics'].values() if isinstance(value,(int,float)))
            totals['decisions']+=16;totals['ticks']+=s['ticks'];totals['terminals']+=len(s['terminal_indices']);rounds.append(s);last=s
        base=rounds[1]['resources'];gpu=max(s['resources']['allocated']-base['allocated'] for s in rounds[2:]);rss=max(s['resources']['rss']-base['rss'] for s in rounds[2:])
        assert report['resource_gate']['passed'] and report['resource_gate']['gpu_growth_bytes']==gpu<=512*1024**2 and report['resource_gate']['rss_growth_bytes']==rss<=4*1024**3
        all_rounds.append(rounds)
    for number in range(8):
        reference=all_rounds[0][number]
        for rank in range(8):
            for key in ['before','after','optimizer_hash_before','optimizer_hash_after','optimizer_steps_before','optimizer_steps_after','metrics']:
                assert all_rounds[rank][number][key]==reference[key],(rank,number,key)
    assert totals==dict(decisions=1024,ticks=root['global_ticks'],terminals=root['global_terminals'])
    telemetry=[json.loads(s) for s in (E/'joint03_gpu_telemetry.jsonl').read_text().splitlines()]
    training_events=[json.loads(s) for s in (E/'joint03.log').read_text().splitlines() if s.startswith('{')]
    first_update=next(e for e in training_events if e['event']=='eightgpu_update_started')
    assert telemetry[0]['time']<first_update['time'] and set(telemetry[0]['alive'])=={d['pid'] for d in inventory}
    for item in inventory:
        assert any(any(int(p.split(',')[0])==item['pid'] and p.split(',')[1].strip()==item['uuid'] and int(p.split(',')[2])>4000 for p in t['compute'].splitlines()) for t in telemetry)
    assert telemetry[-1]['alive']==[]
    result=dict(passed=True,verified_files=len(m)+1,terminal_decisions=4,terminal_index=3,terminal_ticks=tr['ticks'],world_size=8,global_decisions=1024,global_ticks=totals['ticks'],global_terminals=totals['terminals'],
        final_actor=root['final_actor'],joint_actor_steps=root['joint_actor_steps'],joint_critic_steps=root['joint_critic_steps'],failed_attempts_preserved=True,telemetry_samples=len(telemetry),telemetry_startup_missing=False)
    with (R/'download_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
