"""Independently replay held-out actions in Godot and check all recorded rewards."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
from resources import PYTHON,GODOT,PROJECT,environment
ROOT=Path(__file__).resolve().parent
def check(task):
    seed,arm=task
    dest=ROOT/'models'/str(seed)/arm
    while not (dest/'result.json').exists():
        if 'Traceback' in (ROOT/'schedule.log').read_text():raise RuntimeError('Training scheduler failed')
        time.sleep(6)
    subprocess.run([PYTHON,ROOT/'verify_game.py','--root',ROOT,'--executable',GODOT,'--project',PROJECT,'--seed',str(seed),'--arm',arm],check=True,env=environment())
    print('audited',seed,arm,flush=True)
if __name__=='__main__':
    spec=json.loads((ROOT/'design.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check,[(s,a) for a in spec['arms'] for s in spec['training_seeds']]))
    subprocess.run([PYTHON,ROOT/'analyze.py'],env=environment(),check=True)
    print('analysis complete',flush=True)
