import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
r=Path(__file__).resolve().parent;job=dict(passed=False,complete=False,stages=[],started_utc=datetime.now(timezone.utc).isoformat())
def save():
    p=r/'preflight_job.partial.json';p.write_text(json.dumps(job,indent=2));p.replace(r/'preflight_job.json')
commands=[('gpu_tests',[sys.executable,str(r/'gpu_tests.py')]),('gpu_preflight',[sys.executable,str(r/'gpu_preflight.py')]),('memory_probe',[sys.executable,str(r/'memory_probe.py')]),
    ('register',[sys.executable,'-m','tools.mea_shuffle_campaign','register','--root',str(r/'experiment01'),'--source',str(r/'data'),
    '--pin','fe86f1e84f4b3239f2f02a7e5fb140503cfac02e8b4bc0587f76e0fbf753ff61','--model-config',str(r/'model_config.json'),
    '--project',str(r/'project'),'--godot','/home/researcher/mea-r2-22554/runtime_20261116_01/assets/godot/Godot_v4.7.2-stable_linux.x86_64'])]
try:
    for name,command in commands:
        job['current_stage']=name;save();start=time.monotonic()
        with (r/(name+'_stage.log')).open('x') as log:
            p=subprocess.run(command,cwd=r/'project',env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        job['stages'].append(dict(name=name,exit_code=p.returncode,seconds=time.monotonic()-start));save()
        if p.returncode:raise RuntimeError(name+' failed')
    job.update(complete=True,passed=True)
except Exception as exc:job.update(complete=True,passed=False,error=repr(exc))
job['finished_utc']=datetime.now(timezone.utc).isoformat();save()
