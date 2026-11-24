import json,os,sys,subprocess,time
from pathlib import Path
from resources import PYTHON,environment
folder=Path(__file__).resolve().parent
jobs=list(json.loads((folder/'design.json').read_text())['seeds'])
active={}
while True:
    first=(folder.parent/'run.log').read_text()
    if len([s for s in first.splitlines() if s.startswith('start ')])>=8:break
    if 'Traceback' in first:raise RuntimeError('Primary scheduler requires attention')
    time.sleep(5)
while active or jobs:
    for gpu,p in list(active.items()):
        if p.poll() is None:continue
        if p.returncode:raise RuntimeError((p.args,p.returncode))
        del active[gpu]
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    primary={}
    for line in (folder.parent/'run.log').read_text().splitlines():
        if line.startswith('start '):
            _,active_seed,gpu,_=line.split();primary[int(gpu)]=active_seed
    blocked={gpu for gpu,seed in primary.items() if not (folder.parent/'runs'/seed/'control_masked.json').exists()}
    for g in data.splitlines():
        index,free=map(int,g.split(','))
        if not jobs or index in active or index in blocked or free<11500:continue
        seed=jobs.pop(0)
        with (folder/(str(seed)+'.log')).open('a') as log:
            active[index]=subprocess.Popen([PYTHON,'-u',folder.parent/'freeze.py',str(seed),'--severe'],env=dict(environment(),CUDA_VISIBLE_DEVICES=str(index)),stdout=log,stderr=subprocess.STDOUT)
        print('start',seed,index,flush=True)
    time.sleep(5)
print('stress completed',flush=True)
