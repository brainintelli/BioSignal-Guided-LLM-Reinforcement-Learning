"""Schedule the registered offline PPO runs; replay their held-out actions in Godot."""
import json
import os
from pathlib import Path
import subprocess
import time
from resources import environment,PYTHON,GODOT,PROJECT
HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'design.json').read_text())
todo=[(seed,arm) for arm in spec['arms'] for seed in spec['training_seeds']]
active={};failed={}
while todo or active:
    for gpu,(job,proc,log) in list(active.items()):
        if proc.poll() is None:continue
        log.close();del active[gpu]
        seed,arm=job
        if proc.returncode:
            if 'out of memory' not in Path(log.name).read_text(errors='replace').lower():raise RuntimeError((job,gpu,proc.returncode))
            attempt=failed.get(job,0)+1
            if attempt>=4:raise RuntimeError(('GPU allocation failed',job))
            failed[job]=attempt
            partial=HERE/'models'/str(seed)/arm
            if partial.exists():partial.rename(partial.with_name(f'{arm}.oom-{attempt}'))
            todo.append(job);continue
        print('completed',seed,arm,gpu,flush=True)
    if todo:
        listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,free in sorted((tuple(map(int,row.split(','))) for row in listing.splitlines()),key=lambda row:-row[1]):
            if gpu in active:continue
            job=next((j for j in todo if free>=13850+1600*failed.get(j,0)),None)
            if job is None:continue
            todo.remove(job);seed,arm=job
            log=(HERE/f'{seed}-{arm}-{failed.get(job,0)}.log').open('x')
            cmd=[PYTHON,'-u',HERE/'fixed_context_ppo.py','--root',HERE,'--project',PROJECT,'--godot',GODOT,
                 '--model-config',HERE/'model_config.json','--seed',str(seed),'--arm',arm]
            proc=subprocess.Popen(cmd,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(job,proc,log);print('start',seed,arm,gpu,free,flush=True)
    time.sleep(5)
print('all models complete',flush=True)
