"""Schedule the preassigned paired runs on free training GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
BASE=Path.home()/'mea-state2-r2-20261118/run02'
ASSETS=Path.home()/'mea-r2-22554/runtime_20261116_01/assets'
PROJECT=Path.home()/'cpp28080-crossover-20261118-replay/start-scene-paired/game'
PYTHON=ASSETS/'python/bin/python'
GODOT=ASSETS/'godot/Godot_v4.7.2-stable_linux.x86_64'
def environment():
    env=dict(os.environ)
    env.update(json.loads((BASE/'environment.json').read_text()))
    pkg=Path.home()/'mea-predictability-check-369c6af196e0'
    env['PYTHONPATH']=':'.join(map(str,[pkg,pkg/'mea_binary_r2',pkg/'mea_state2_r2']))+':'+env['PYTHONPATH']
    return env

def main():
    study=json.loads((ROOT/'design.json').read_text())
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus=[int(line.split(',')[0]) for line in info.splitlines() if int(line.split(',')[1])>20000]
    if not gpus:
        raise RuntimeError('No available GPU')
    jobs=[(s,a) for s in study['training_seeds'] for a in study['arms']]
    folders=ROOT/'models'
    folders.mkdir(exist_ok=True)
    def run_queue(slot):
        for seed,arm in jobs[slot::len(gpus)]:
            dest=folders/str(seed)
            dest.mkdir(exist_ok=True)
            if (dest/arm/'result.json').exists():
                continue
            cmd=[PYTHON,'-u',ROOT/'fixed_context_ppo.py','--root',ROOT,'--project',PROJECT,'--godot',GODOT,
                 '--model-config',BASE/'model_config.json','--seed',str(seed),'--arm',arm]
            with (dest/(arm+'.log')).open('x') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpus[slot])),check=True)
            print('completed',seed,arm,flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_queue,range(len(gpus))))
    print('completed all models',flush=True)

if __name__=='__main__':
    main()
