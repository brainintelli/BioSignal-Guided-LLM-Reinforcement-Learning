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

