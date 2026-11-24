"""Read-only campaign verification, then archive; never runs models or devices."""
import hashlib,json,sys,zipfile
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117');project=r/'project';root=r/'experiment01'
sys.path.insert(0,str(project))
from tools.mea_p0 import sha,read_json,_same
from tools import mea_stim_train as t
from tools.mea_stim_campaign import registration
from agent.mea_stim_offline import PROTOCOL,analyze,validate_budget
from agent.mea_protocol import digest
job=read_json(r/'job.json');status=read_json(root/'batch_status.json');saved=read_json(root/'analysis.json')
if job.get('accepted_complete') is not True or status.get('passed') is not True or not status.get('complete') or saved.get('status')!='complete':raise RuntimeError('Campaign not accepted complete')
for key in ('worker_pid','child_pid'):
    if Path('/proc')/str(job[key]) in Path('/proc').iterdir():raise RuntimeError('Campaign process still present')
pin='fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61'
reg=registration(root,r/'data',pin,r/'model_config.json',project,Path('/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64'))
_same(digest(reg),'b520f2e1ef88048c2735fba2fc3a6a6ad3b31abf870c9245c753462d406290a1','original registration pin')
if len(status['cells'])!=24 or any(len(c['stages'])!=3 or any(s['exit_code']!=0 for s in c['stages']) for c in status['cells']):raise RuntimeError('Missing successful stages')
cells=[];records=0
for b in range(8):
    for arm in PROTOCOL['arms']:
        out=root/f'b{b:02d}_{arm}';q=read_json(root/f'verified_b{b:02d}_{arm}.json')
        for k,v in dict(passed=True,block=b,arm=arm,registration_sha256=digest(reg),source_pin=pin,training_decisions=384,fresh_process=True).items():_same(q[k],v,'receipt '+k)
        _same(q['cell_manifest_sha256'],sha(out/'cell_manifest.json'),'cell manifest');t.check_manifest(out,'cell_manifest.json')
        _same(q['training_receipt_sha256'],sha(out/'train/independent_verification.json'),'training receipt')
        if set(q['eval_errors'])!={'pre_before','pre_after','post_before','post_after'} or any(not 0<=x<=1e-6 for x in q['eval_errors'].values()):raise RuntimeError('Evaluation errors')
        training=read_json(out/'train/report.json');validate_budget(training['decisions'],[x['applied_steps'] for x in training['rounds']])
        _same(training['initial'],reg['common_initial'],'initial pin')
        evaluation=read_json(out/'evaluation.json');_same(evaluation['checkpoint_tensors']['pre'],reg['common_initial'],'pre eval pin');_same(evaluation['checkpoint_tensors']['post'],training['final'],'post eval pin')
        cells.append(dict(block=b,arm=arm,**{p:{c:evaluation['metrics'][p][c]['discounted_r1'] for c in ('before','after')} for p in ('pre','post')}))
        for p in out.rglob('rollout.jsonl'):
            records+=sum(1 for line in p.open() if line.strip())
recomputed=analyze(cells)|dict(cells=cells,registration_sha256=digest(reg),source_pin=pin)
_same(saved,recomputed,'recomputed final analysis')
for name,h in read_json(r/'project_source_manifest.json').items():_same(sha(project/name),h,'final frozen source')
names=['project_frozen.zip','project_source_manifest.json','model_config.json','preflight_passed.json','gpu_preflight_tests.log','launch.json','job.json','batch.log','worker.log','worker.py','launch_once.py']
files=[r/n for n in names]+[p for parent in [root,r/'data'] for p in sorted(parent.rglob('*')) if p.is_file()]
manifest={p.relative_to(r).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
accepted=dict(passed=True,files=manifest,records=records,cells=24,training_decisions=9216,registration_sha256=digest(reg),source_pin=pin,analysis=saved)
path=r/'accepted_campaign_manifest.json'
with path.open('x') as f:json.dump(accepted,f,ensure_ascii=False,indent=2)
files.append(path);archive=r/'campaign_delivery.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for p in files:z.write(p,p.relative_to(r).as_posix())
receipt=dict(passed=True,files=len(files),records=records,bytes=archive.stat().st_size,archive_sha256=sha(archive),accepted_manifest_sha256=sha(path))
with (r/'archive_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
