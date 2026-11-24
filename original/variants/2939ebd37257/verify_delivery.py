"""Verify downloaded evidence and both complete campaigns; writes receipt once."""
import hashlib,json,math
from pathlib import Path
from protocol import audit_gae,optimizer_gate,resource_gate
ROOT=Path(__file__).resolve().parent;E=ROOT/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def verify():
    receipt={};total_files=set()
    for name,entry in read(E/'evidence_manifest.json').items():
        p=(E/name).resolve();assert p.is_relative_to(E.resolve()) and p.stat().st_size==entry['size'] and sha(p)==entry['sha256'],name
    runtime=read(E/'runtime_audit.json')
    assert runtime['passed'] and runtime['files']==24551 and runtime['manifest_sha256']=='3f71b79f474aa61d4fa7ebd35d0b07f90b2988ddc01e110aeddb8ca7fed82737'
    for name in ['stability_run.py','protocol.py','test_protocol.py','probe_terminal_fixture.py','package_evidence.py']:
        assert sha(ROOT/name)==sha(E/name),name
    old=ROOT.parent/'mea_22554_gpu_smoke_20261116'
    assert sha(old/'checkpoint_smoke.py')==sha(E/'base_smoke.py')
    assert sha(old/'layout_offload.py')==sha(E/'layout_offload.py')
    for phase,dirname,n,round_count in [('terminal','terminal01',32,1),('stability','stability01',128,8)]:
        out=E/dirname;report=read(out/'report.json');ind=read(out/'independent_verification.json');identity=read(out/'identity.json')
        assert (E/(dirname+'.exit')).read_text().strip()=='0'
        assert (E/(dirname+'_verify.exit')).read_text().strip()=='0'
        assert report['passed'] and ind['passed'] and report['phase']==phase and report['rounds']==round_count
        assert report['decisions']==n*round_count and ind['collection_pid']==report['collection_pid']!=ind['verification_pid']
        assert ind['campaign_manifest_sha256']==sha(out/'campaign_manifest.json')
        assert report['alpha']==1 and not any(report[k] for k in ['stimulation','r2_enabled','state2_enabled'])
        assert report['fresh_destination_rollout'] and not report['cross_host_probe_passed']
        if phase=='stability':
            env=read(E/'execution_environment.json');assert env['hostname']=='researcher-local-6'
            process=next(p for p in env['processes'] if p['pid']==report['collection_pid'])
            assert process['environment']['CUDA_VISIBLE_DEVICES']=='GPU-e9739564-e2ec-59c3-10f7-096d780abcbc'
            assert f'{report["collection_pid"]}, GPU-e9739564-e2ec-59c3-10f7-096d780abcbc,' in env['gpu_processes']
        assert identity['driver']==sha(E/'stability_run.py') and identity['protocol_module']==sha(E/'protocol.py')
        assert identity['fixture']==sha(E/'terminal_fixture.json')
        assert identity['base']==read(old/'evidence/run05/source_identity.json')
        assert identity['config']==read(old/'evidence/run05/config.json')
        for name,h in read(out/'campaign_manifest.json').items():
            p=(out/name).resolve();assert p.is_relative_to(out.resolve()) and sha(p)==h,name;total_files.add(p)
        summaries=[];last=None;terminals=0;ticks=0
        for number in range(1,round_count+1):
            rd=out/f'round_{number:02d}';s=read(rd/'report.json');v=ind['rounds'][number-1];boot=read(rd/'bootstrap.json')
            rows=[json.loads(line)['training'] for line in (rd/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(rows)==n and s['decisions']==n and v['decisions']==n
            assert all(s['checks'].values()) and v['native_replay'] and v['input_identity'] and v['gae_exact']
            for x in [s['pre_update_statistic_error'],s['roundtrip_error'],v['old_error'],v['new_error'],v['bootstrap_error']]:assert math.isfinite(x) and x<=1e-6
            gae=audit_gae(rows,boot['value'],identity['config']['gamma_tick'],identity['config']['gae_lambda'])
            assert gae['terminal_indices']==s['terminal_indices']==v['terminal_indices']
            if phase=='terminal':assert any(i<n-1 for i in gae['terminal_indices'])
            if last:
                assert s['before']==last['after'] and s['optimizer_steps_before']==last['optimizer_steps_after']
            else:
                assert s['before']['actor']=='94d7dd9adaa86ecaffacb8dbec584054ba96091c816da3bb1bf59db8249df5e0'
                assert s['optimizer_steps_before']==dict(actor=[],critic=[])
            optimizer_gate(s['optimizer_steps_before'],s['optimizer_steps_after'],s['applied_steps']['actor'],s['applied_steps']['critic'])
            assert s['optimizer_steps_after']==v['optimizer_steps_after']
            ticks+=s['ticks'];terminals+=len(s['terminal_indices']);summaries.append(s);last=s
        assert terminals==report['terminals'] and report['final_actor']==last['after']['actor']
        if phase=='stability':assert resource_gate([s['resources'] for s in summaries])==report['resource_gate'] and report['resource_gate']['passed']
        receipt[phase]=dict(decisions=n*round_count,rounds=round_count,ticks=ticks,terminals=terminals,passed=True)
    receipt['verified_manifest_files']=len(total_files)
    with (ROOT/'download_verification.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))
if __name__=='__main__':verify()
