from pathlib import Path
import json
import os
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from resources import PYTHON,environment

def main():
    s=json.loads((HERE/'design.json').read_text());queue=[(seed,arm) for seed in s['seeds'] for arm in s['arms']]
    active={};last_start={};retries={}
    while queue or active:
        for pid,(proc,log,job,gpu) in list(active.items()):
            if proc.poll() is None:continue
            log.close();del active[pid]
            if proc.returncode:
                if 'out of memory' not in Path(log.name).read_text(errors='replace').lower():raise RuntimeError((job,gpu,proc.returncode))
                retries[job]=retries.get(job,0)+1
                if retries[job]>2:raise RuntimeError(('allocation failed',job))
                queue.append(job)
            else:print('complete',job,gpu,flush=True)
        if queue:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            slots=sorted((tuple(map(int,x.split(','))) for x in raw.splitlines()),key=lambda x:-x[1])
            for gpu,free in slots:
                if not queue:break
                occupying=sum(g==gpu for _,_,_,g in active.values());job=queue[0]
                if free < (11000 if retries.get(job) else 9600) or occupying>=2 or time.monotonic()-last_start.get(gpu,0)<35:continue
                seed,arm=queue.pop(0)
                log=(HERE/f'{seed}-{arm}-{retries.get(job,0)}.log').open('a')
                proc=subprocess.Popen([PYTHON,'-u',HERE/'evaluate.py',str(seed),arm],env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                active[proc.pid]=(proc,log,job,gpu);last_start[gpu]=time.monotonic();print('start',job,gpu,free,flush=True)
        time.sleep(4)
    print('all complete',flush=True)
if __name__=='__main__':main()
