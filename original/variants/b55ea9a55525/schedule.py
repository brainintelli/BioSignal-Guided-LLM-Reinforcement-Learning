"""Run each registered seed on an available GPU and retain the per-seed logs."""
from pathlib import Path
import json
import os
import subprocess
import time

from resources import PYTHON,environment

HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'design.json').read_text())
queue=list(spec['seeds'])
active={}
failed_gpus={seed:set() for seed in queue}
while active or queue:
    for gpu,(seed,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[gpu]
        if p.returncode==0:continue
        message=Path(log.name).read_text(errors='replace')
        if 'OutOfMemoryError' not in message and 'out of memory' not in message.lower():raise RuntimeError((seed,gpu,p.returncode))
        failed_gpus[seed].add(gpu)
        if len(failed_gpus[seed])>=8:raise RuntimeError(('out of GPU memory',seed))
        queue.append(seed)
    if queue:
        listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,free in sorted((tuple(map(int,row.split(','))) for row in listing.splitlines()),key=lambda p:-p[1]):
            seed=next((s for s in queue if gpu not in failed_gpus[s]),None)
            if seed is None or free<8800 or gpu in active:continue
            queue.remove(seed);log=(HERE/f'{seed}-gpu{gpu}.log').open('a')
            p=subprocess.Popen([PYTHON,'-u',HERE/'run.py',str(seed)],env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(seed,p,log);print('started',seed,gpu,free,flush=True)
    time.sleep(4)
print('complete',flush=True)
