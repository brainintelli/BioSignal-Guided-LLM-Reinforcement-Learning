import hashlib,json,os,subprocess,sys
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117');project=r/'project'
if any((r/name).exists() for name in ['launch.json','job.json','batch.log','experiment01/batch_status.json']):raise RuntimeError('Already launched')
if json.loads((r/'preflight_passed.json').read_text())['passed'] is not True:raise RuntimeError('Preflight missing')
for n,h in json.loads((r/'project_source_manifest.json').read_text()).items():
    if hashlib.sha256((project/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed')
sys.path.insert(0,str(project));from agent.mea_protocol import digest
reg=json.loads((r/'experiment01/registration.json').read_text())
if digest(reg)!='b520f2e1ef88048c2735fba2fc3a6a6ad3b31abf870c9245c753462d406290a1':raise RuntimeError('Registration changed')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if gpu:raise RuntimeError('GPU occupied; do not compete with another job')
with (r/'worker.log').open('x') as log:
    p=subprocess.Popen([sys.executable,str(r/'worker.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,automatic_restart=False,registered_cells=24,alpha=1,r2_mode='record_only',realtime_training=False)
with (r/'launch.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
