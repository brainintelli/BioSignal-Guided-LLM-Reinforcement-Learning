import concurrent.futures
import json
from pathlib import Path
import subprocess
import time
from launch import ROOT,GODOT,PROJECT,PYTHON,environment

def audit(key):
    s,a=key
    path=ROOT/'models'/str(s)/a
    while not (path/'result.json').exists():
        time.sleep(8)
    cmd=[PYTHON,ROOT/'verify_game.py','--root',ROOT,'--executable',GODOT,'--project',PROJECT,'--seed',str(s),'--arm',a]
    with (path/'audit.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment(),check=True)
    print('verified',s,a,flush=True)

if __name__=='__main__':
    spec=json.loads((ROOT/'design.json').read_text())
    jobs=[(s,a) for s in spec['training_seeds'] for a in spec['arms']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(audit,jobs))
