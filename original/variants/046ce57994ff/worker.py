import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parent;project=r/'project';root=r/'experiment01'
job={'worker_pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'automatic_restart':False,'accepted_complete':False}
def persist():
    p=r/'job.partial.json';p.write_text(json.dumps(job,indent=2));p.replace(r/'job.json')
cmd=[sys.executable,'-u','-m','tools.mea_shuffle_parallel','--root',str(root),'--source',str(r/'data'),
    '--model-config',str(r/'model_config.json'),'--project',str(project),
    '--godot','/home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64']
with (r/'parallel.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=project,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    job['child_pid']=child.pid;persist();job['exit_code']=child.wait()
try:
    status=json.loads((root/'parallel_status.json').read_text());analysis=json.loads((root/'analysis.json').read_text())
    job['accepted_complete']=job['exit_code']==0 and status.get('passed') is True and status.get('accepted_cells')==32 and analysis.get('status')=='complete' and len(analysis.get('cells',[]))==32
except Exception as exc:job['error']=repr(exc)
job['finished_utc']=datetime.now(timezone.utc).isoformat();persist()
