"""Resume only verified completed jobs; run at most two inference workers per GPU."""
from pathlib import Path
import json,os,subprocess,sys,time
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from resources import PYTHON,environment
s=json.loads((H/'design.json').read_bytes());arms=s['arms'];scenes=list(s['scenes'])
queue=[]
# Interleave scene, trained seed and arm to distribute load without condition-specific devices.
for k in range(9):
    for j,seed in enumerate(s['training_seeds']):
        arm=arms[(k+j)%3];scene=scenes[(k//3+j)%3]
        if not (H/'games'/str(seed)/scene/arm/'frozen.json').exists():queue.append((seed,arm,scene))
assert len(set(queue))==len(queue)
active={};last={};bad={}
while queue or active:
    for pid,(job,gpu,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[pid]
        if p.returncode:
            msg=Path(log.name).read_text(errors='replace')
            if 'out of memory' not in msg.lower():raise RuntimeError((job,p.returncode,msg[-1600:]))
            hist=bad.setdefault(job,[]);hist.append(gpu)
            if len(hist)>3:raise RuntimeError(('persistent allocation failure',job))
            queue.append(job)
        else:print('ok',job,gpu,flush=True)
    if queue:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,free in sorted((tuple(map(int,x.split(','))) for x in raw.splitlines()),key=lambda z:-z[1]):
            if sum(v[1]==gpu for v in active.values())>=2 or time.monotonic()-last.get(gpu,0)<30:continue
            job=next((j for j in queue if free>=(11600 if gpu in bad.get(j,[]) else 9600)),None)
            if job is None:continue
            queue.remove(job);seed,arm,scene=job
            log=(H/f'{seed}-{arm}-{scene}-{time.time_ns()}.log').open('x')
            p=subprocess.Popen([PYTHON,'-u',H/'evaluate.py',str(seed),arm,scene],stdout=log,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)))
            active[p.pid]=(job,gpu,p,log);last[gpu]=time.monotonic();print('start',job,gpu,free,flush=True)
    time.sleep(3)
print('all complete',flush=True)
