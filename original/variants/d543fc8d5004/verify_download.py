"""Verify authenticated download and replay scalar acceptance independently of GPU code."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,math
from check_acceptance import validate
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
transfer=ROOT/'transfer';receipt=read(transfer/'evidence_receipt.json')
expected='5623a51821726dd89d9849811c95b261'+'9efaafa0e1234fa3bba9539b1bc011ce'
assert receipt['archive_sha256']==sha(transfer/'evidence.tar')==expected
assert (transfer/'evidence.tar').stat().st_size==receipt['archive_size']
dest=ROOT/'evidence'
with tarfile.open(transfer/'evidence.tar') as tar:
    members=tar.getmembers()
    assert len(members)==receipt['file_count']==len(receipt['files'])
    assert {m.name for m in members}==set(receipt['files'])
    for m in members:
        p=PurePosixPath(m.name)
        assert m.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in m.name
        assert (dest/m.name).resolve().is_relative_to(dest.resolve())
    dest.mkdir(exist_ok=False);tar.extractall(dest,filter='data')
for name,entry in receipt['files'].items():
    p=dest/name;assert p.stat().st_size==entry['size'] and sha(p)==entry['sha256'],name
for name in ['checkpoint_smoke.py','layout_offload.py','check_acceptance.py','run_smoke.sh','test_layout_offload.py']:
    assert sha(ROOT/name)==sha(dest/name),name
runtime=read(dest/'runtime/verification.json');rm=read(dest/'runtime/runtime_manifest.json')
assert runtime['all_files_match'] and runtime['verified_files']==len(rm['files'])==24551
assert runtime['manifest_sha256']==sha(dest/'runtime/runtime_manifest.json')==('3f71b79f474aa61d4fa7ebd35d0b07f9'+'0b2988ddc01e110aeddb8ca7fed82737')
original=ROOT.parent/'mea_22554_followup_20261116/model_transfer/bundle_manifest.json'
model=read(original)
for name,entry in model['files'].items():
    prefix='m9-research-v5-project/'
    if name.startswith(prefix):assert sha(dest/'project'/name[len(prefix):])==entry['sha256']
run=dest/'run05';report=read(run/'report.json');independent=read(run/'independent_reload.json')
assert validate(report,independent)
assert read(dest/'run05_exit.json')['exit_code']==0
for name,h in read(run/'collection_manifest.json').items():assert sha(run/name)==h,name
assert independent['collection_manifest_sha256']==sha(run/'collection_manifest.json')
assert independent['harness_sha256']==sha(ROOT/'checkpoint_smoke.py')
assert read(run/'source_identity.json')['asset_manifest']==sha(original)
assert read(run/'source_identity.json')['layout_offload']==sha(ROOT/'layout_offload.py')
cp=read(run/'checkpoint/manifest.json')
for name,h in cp['files'].items():assert sha(run/'checkpoint'/name)==h,name
assert cp['actor_sha256']==report['final_actor']
source=read(dest/'source_probe.json');probe=read(run/'migration_probe.json')
assert source['error']==0 and source['old_probe']==source['recomputed']==probe['saved_source']
assert probe['destination']==probe['destination_reloaded']
error=max(abs(a-b) for a,b in zip(source['recomputed'],probe['destination']))
assert error==report['cross_host_probe_error']==probe['cross_host_error'] and not report['cross_host_probe_passed']
events=[json.loads(x) for x in (run/'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
config=read(run/'config.json');bootstrap=read(run/'bootstrap.json');rows=[e['training'] for e in events]
assert len(rows)==report['decisions']==32
assert config['alpha']==1 and not config['r2_enabled'] and not config['state2_enabled']
assert config['activation_offload']==report['activation_offload']=='cpu_pinned_preserve_strides'
assert sum(r['duration'] for r in rows)==report['ticks']==40
for i,(e,r) in enumerate(zip(events,rows)):
    assert e['index']==i and e['prepared']['branches'][r['verb']][r['target']]==e['action']
    assert r['reward']==e['r1_score']['reward']==e['reward_record']['r1']==e['reward_record']['total_reward']
    assert e['reward_record']==dict(alpha=1.,r1=r['reward'],total_reward=r['reward'],r2_enabled=False,r2_valid=False,r2_raw=None)
    assert r['boundary']==(r['terminal'] or i==len(rows)-1)
    if i+1<len(rows):assert e['next_observation']==events[i+1]['observation'] and not r['terminal']
carry=0.;max_error=0.
for i in reversed(range(len(rows))):
    r=rows[i];n=0. if r['terminal'] else rows[i+1]['value'] if i+1<len(rows) else bootstrap['value']
    d=config['gamma_tick']**r['duration']
    carry=r['reward']+d*n-r['value']+d*config['gae_lambda']**r['duration']*(not r['boundary'])*carry
    max_error=max(max_error,abs(carry-r['advantage']),abs(carry+r['value']-r['return']))
assert max_error<1e-12
release=read(dest/'gpu_release.json');assert release['selected_gpu_no_compute_process']
result=dict(all_passed=True,archive_files_verified=receipt['file_count'],archive_sha256=expected,
    runtime_source_manifest_verified=True,original_project_snapshot_verified=True,decisions=32,native_replay_remote_passed=True,
    independent_local_gae_max_error=max_error,reward_exact_matches=32,final_actor=report['final_actor'],
    cross_host_probe_passed=False,cross_host_probe_error=error,local_reload_passed=True,selected_gpu_released=True,
    verifier_sha256=sha(Path(__file__)))
with (ROOT/'download_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
