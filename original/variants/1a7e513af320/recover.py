"""Retry one resource failure from the original warm start, preserving its partial artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from launch import ASSETS,BASE,GODOT,PROJECT,PYTHON,ROOT,environment

def recover(seed,arm):
    spec=json.loads((ROOT/'design.json').read_text())
    if seed not in spec['training_seeds'] or arm not in spec['arms']:
        raise ValueError('Unknown training assignment')
    folder=ROOT/'models'/str(seed)/arm
    log=folder.with_suffix('.log')
    if (folder/'result.json').exists() or 'torch.OutOfMemoryError' not in log.read_text():
        raise ValueError('Only a recorded, incomplete GPU OOM can enter this retry')
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=proc.read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if str(ROOT/'fixed_context_ppo.py') in argv and '--seed' in argv and argv[argv.index('--seed')+1]==str(seed) and argv[argv.index('--arm')+1]==arm:
            raise RuntimeError('Failed worker has not exited')
    archive=ROOT/'failed-attempts'/(str(seed)+'-'+arm)
    if folder.resolve().parent.parent!=ROOT.resolve()/'models' or archive.exists():
        raise ValueError('Unexpected source or archive location')
    archive.mkdir(parents=True)
    folder.rename(archive/'partial')
    log.rename(archive/'original.log')
    while True:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free=[(int(line.split(',')[1]),int(line.split(',')[0])) for line in out.splitlines() if int(line.split(',')[0])!=0]
        mem,gpu=max(free)
        if mem>=14500:break
        time.sleep(10)
    print('restart',seed,arm,'GPU',gpu,'free MiB',mem,flush=True)
    cmd=[PYTHON,'-u',ROOT/'fixed_context_ppo.py','--root',ROOT,'--project',PROJECT,'--godot',GODOT,
         '--model-config',BASE/'model_config.json','--seed',str(seed),'--arm',arm]
    with log.open('x') as stream:
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)),check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('arm');a=p.parse_args();recover(a.seed,a.arm)
