"""Run one fresh fit or one frozen evaluation at a time per available GPU."""
from pathlib import Path
import json
import os
import subprocess
import time
from resources import PYTHON,GODOT,PROJECT,environment
HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'design.json').read_text())
queue=[]
for j,seed in enumerate(spec['training_seeds']):
    for i in range(3):
        arm=spec['arms'][(i+j)%3]
        if not (HERE/'games'/str(seed)/arm/'frozen.json').exists():
            stage='eval' if (HERE/'models'/str(seed)/arm/'update.json').exists() else 'fit'
            queue.append((seed,arm,stage))
active={}
bad={}
while queue or active:
    for gpu,(job,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[gpu]
        seed,arm,stage=job
        if p.returncode:
            msg=Path(log.name).read_text(errors='replace')
            if 'out of memory' not in msg.lower():raise RuntimeError((job,gpu,msg[-900:]))
            bad.setdefault(job,[]).append(gpu)
            if len(bad[job])>3:raise RuntimeError(('repeated OOM',job))
            if stage=='fit':
                f=HERE/'models'/str(seed)/arm
                if f.exists():f.rename(f.with_name(arm+'.oom-'+str(time.time_ns())))
            queue.append(job)
        else:
            print('ok',job,gpu,flush=True)
            if stage=='fit':queue.append((seed,arm,'eval'))
    if queue:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,free in sorted((tuple(map(int,l.split(','))) for l in raw.splitlines()),key=lambda v:-v[1]):
            if gpu in active:continue
            job=next((j for j in queue if free>=(13800 if j[2]=='fit' else 9550)+(1800 if gpu in bad.get(j,[]) else 0)),None)
            if job is None:continue
            queue.remove(job);seed,arm,stage=job
            if stage=='fit':
                cmd=[PYTHON,'-u',HERE/'train.py','--root',HERE,'--godot',GODOT,'--project',PROJECT,'--model-config',HERE/'model_config.json','--seed',str(seed),'--arm',arm]
            else:cmd=[PYTHON,'-u',HERE/'play60.py',str(seed),arm]
            log=(HERE/f'{seed}-{arm}-{stage}-{time.time_ns()}.log').open('x')
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)))
            active[gpu]=(job,proc,log);print('start',job,gpu,free,flush=True)
    time.sleep(3)
print('all complete',flush=True)
