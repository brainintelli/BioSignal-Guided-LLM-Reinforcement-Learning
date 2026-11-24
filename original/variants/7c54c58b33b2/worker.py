import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117');project=r/'project';root=r/'experiment01'
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',PYTHONPATH='/home/researcher/meadowmind-gpu/m8b-runtime-candidates/causal_conv1d_1_7_0_sourcebuild/site:'+str(project))
cmd=[sys.executable,'-u','-m','tools.mea_stim_campaign','batch','--root',str(root),'--source',str(r/'data'),
 '--pin','fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61','--model-config',str(r/'model_config.json'),
 '--project',str(project),'--godot','/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64']
job=dict(worker_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),automatic_restart=False)
def persist():
    p=r/'job.partial.json';p.write_text(json.dumps(job,indent=2));p.replace(r/'job.json')
with (r/'batch.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    job['child_pid']=child.pid;persist();job['exit_code']=child.wait()
job['finished_utc']=datetime.now(timezone.utc).isoformat()
try:
    status=json.loads((root/'batch_status.json').read_text());analysis=json.loads((root/'analysis.json').read_text())
    job['accepted_complete']=job['exit_code']==0 and status.get('passed') is True and status.get('complete') is True and analysis.get('status')=='complete'
except (OSError,ValueError):job['accepted_complete']=False
persist()
