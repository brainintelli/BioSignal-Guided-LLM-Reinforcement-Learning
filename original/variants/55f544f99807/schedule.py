"""Dispatch independent inference workers and retain completed game receipts across retries."""
from pathlib import Path
import os,sys,time,json,subprocess
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from resources import PYTHON,environment
s=json.loads((HERE/'design.json').read_bytes());q=[]
conds=list(s['conditions'])
for j,seed in enumerate(s['training_seeds']):
    for i in range(len(conds)):
        c=conds[(i+j)%len(conds)]
        if not (HERE/'games'/str(seed)/c/'frozen.json').exists():q.append((seed,c))
active={};fail={}
while q or active:
    for gpu,(job,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[gpu]
        if p.returncode:
            msg=Path(log.name).read_text(errors='replace');tries=fail.setdefault(job,[]);tries.append(gpu)
            if 'out of memory' not in msg.lower() or len(tries)>3:raise RuntimeError((job,p.returncode,msg[-1800:]))
            q.append(job)
        else:print('ok',job,gpu,flush=True)
    if q:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,mem in sorted((tuple(map(int,v.split(','))) for v in free.splitlines()),key=lambda x:-x[1]):
            if gpu in active:continue
            job=next((j for j in q if mem>=(11100 if gpu in fail.get(j,[]) else 9500)),None)
            if job is None:continue
            q.remove(job);seed,c=job;log=(HERE/f'{seed}-{c}-{time.time_ns()}.log').open('x')
            p=subprocess.Popen([PYTHON,'-u',HERE/'evaluate.py',str(seed),c],stdout=log,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)))
            active[gpu]=(job,p,log);print('start',job,gpu,mem,flush=True)
    time.sleep(3)
print('all complete',flush=True)
