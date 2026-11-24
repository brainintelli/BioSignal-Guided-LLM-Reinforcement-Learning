"""Run the prespecified game-only control from two matched pairs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parent
base=Path.home()/'mea-state2-r2-20261118/run02'
assets=Path.home()/'mea-r2-22554/runtime_20261116_01/assets'
python=assets/'python/bin/python'
godot=assets/'godot/Godot_v4.7.2-stable_linux.x86_64'
study=json.loads((root/'design.json').read_text())
env=dict(os.environ)
env.update(json.loads((base/'environment.json').read_text()))
pkg=Path.home()/'mea-predictability-check-369c6af196e0'
env['PYTHONPATH']=':'.join(map(str,[pkg,pkg/'mea_binary_r2',pkg/'mea_state2_r2']))+':'+env['PYTHONPATH']
project=root/'game'
if not project.exists():
    project.mkdir()
    for folder in ('configs','scripts','godot_tests'):
        shutil.copytree(base/'project'/folder,project/folder)
    shutil.copy2(base/'project/project.godot',project/'project.godot')
    custom=root/'supplied_session.gd'
    bridge=project/'godot_tests/m9_environment.gd'
    if hashlib.sha256(custom.read_bytes()).hexdigest()!=study['variant_sha256'] or hashlib.sha256(bridge.read_bytes()).hexdigest()!=study['bridge_base_sha256']:
        raise ValueError('Unregistered game sources')
    before='res://scripts/m9_environment_session.gd'
    source=bridge.read_text()
    if source.count(before)!=1:
        raise ValueError('Expected exactly one session import')
    shutil.copy2(custom,project/'scripts/mea_supplied_session.gd')
    bridge.write_text(source.replace(before,'res://scripts/mea_supplied_session.gd'))
subprocess.run([python,root/'pilot.py',root,godot,project],env=env,check=True)
info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free=[int(s.split(',')[0]) for s in info.splitlines() if int(s.split(',')[1])>20000]
if not free:
    raise RuntimeError('Training GPUs busy')
assignments=[(s,a) for s in study['training_seeds'] for a in study['arms']]
attempt=root/'models'
attempt.mkdir(exist_ok=False)
for start in range(0,len(assignments),len(free)):
    jobs={}
    for gpu,(seed,arm) in zip(free,assignments[start:]):
        dest=attempt/str(seed)/arm
        dest.parent.mkdir(exist_ok=True)
        cmd=[python,'-u',root/'crossover_replay_ppo.py','--model-config',base/'model_config.json',
             '--godot',godot,'--project',project,'--data',root/'ppo-data.json','--study',root/'design.json',
             '--train-seed',str(seed),'--arm',arm,'--out',dest]
        with dest.with_suffix('.log').open('x') as log:
            jobs[(seed,arm)]=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)))
    rc={str(key):job.wait() for key,job in jobs.items()}
    print(rc,flush=True)
    if any(rc.values()):
        sys.exit(1)
