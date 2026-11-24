"""Schedule paired offline training on the same severe-start game for both conditions."""
from pathlib import Path
import json
import os
import subprocess
import time
from resources import PYTHON,GODOT,environment
HERE=Path(__file__).resolve().parent
GAME=Path.home()/'mea-22021-20261118-mea-stim-ablation/archive-frozen-60tick/severe/plain-game'
s=json.loads((HERE/'design.json').read_text());queue=[(seed,arm) for seed in s['training_seeds'] for arm in s['arms']]
queue=[j for j in queue if not (HERE/'models'/str(j[0])/j[1]/'update.json').is_file()]
active={};tries={};bad={}
for seed,arm in queue:
    existing=list((HERE/'models'/str(seed)).glob(arm+'.oom-*'))
    if existing:
        tries[seed,arm]=len(existing)
        starts=(HERE/'training.log').read_text().splitlines()
        bad[seed,arm]={int(line.split()[-2]) for line in starts if line.startswith('start '+str((seed,arm))+' ')}
while queue or active:
    for gpu,(job,p,log) in list(active.items()):
        if p.poll() is None:continue
        log.close();del active[gpu]
        if p.returncode:
            if 'out of memory' not in Path(log.name).read_text(errors='replace').lower():raise RuntimeError((job,gpu,p.returncode))
            count=tries.get(job,0)+1;tries[job]=count;bad.setdefault(job,set()).add(gpu)
            if count>3:raise RuntimeError(('allocation failed',job))
            folder=HERE/'models'/str(job[0])/job[1]
            if folder.exists():folder.rename(folder.with_name(folder.name+'.oom-'+str(count)))
            queue.append(job)
        else:print('complete',job,gpu,flush=True)
    if queue:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,free in sorted((tuple(map(int,l.split(','))) for l in raw.splitlines()),key=lambda x:-x[1]):
            if gpu in active:continue
            job=next((j for j in queue if free>=(16000 if gpu in bad.get(j,set()) else 13800)),None)
            if not job:continue
            seed,arm=job;queue.remove(job)
            log=(HERE/f'{seed}-{arm}-{tries.get(job,0)}.log').open('x')
            cmd=[PYTHON,'-u',HERE/'fixed_context_ppo.py','--root',HERE,'--godot',GODOT,'--project',GAME,
                 '--model-config',HERE/'model_config.json','--seed',str(seed),'--arm',arm,'--train-only']
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)))
            active[gpu]=(job,proc,log);print('start',job,gpu,free,flush=True)
    time.sleep(4)
print('all complete',flush=True)
