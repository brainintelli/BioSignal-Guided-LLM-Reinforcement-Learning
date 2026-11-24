"""Node-5 regressions with real exit codes and engine-error checks."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path('/home/researcher/meadowmind-gpu/m8b-project')
folder = Path('/home/researcher/meadowmind-gpu/diagnostics/rl_speed_20261107_regression')
folder.mkdir(exist_ok=False)
(root/'output/runtime_tests').mkdir(parents=True,exist_ok=True)
godot = '/home/researcher/meadowmind-gpu/godot/Godot_v4.7.2-stable_linux.x86_64'
os.environ.update(MEADOW_GODOT=godot, PYTHONDONTWRITEBYTECODE='1', GODOT_SILENCE_ROOT_WARNING='1')
checks = []
def run(name, command):
    begin = time.monotonic()
    with (folder/(name+'.log')).open('w') as stream:
        result = subprocess.run(command,cwd=root,env=os.environ,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    log = (folder/(name+'.log')).read_text(errors='replace')
    bad = bool(re.search(r'(?m)(SCRIPT ERROR:|ERROR:|FAIL failures=|FAILED \()', log))
    row = dict(name=name, passed=result.returncode==0 and not bad, exit_code=result.returncode,
               seconds=time.monotonic()-begin, log=name+'.log')
    checks.append(row)
    print(json.dumps(row),flush=True)
    (folder/'checks.json').write_text(json.dumps(checks,indent=2))

run('import',[godot,'--headless','--path',str(root),'--editor','--import'])
run('python',[sys.executable,'-m','unittest','discover','-s','tests','-v'])
for name in ('test_world','test_controller','test_collision','test_collision_physics','test_survival_ui',
             'test_direction_animation','test_rl_environment','test_m8b_game_entry','test_m8x_game_entry','test_model_session'):
    command=[godot,'--headless','--path',str(root),'--script','godot_tests/'+name+'.gd']
    if name=='test_m8b_game_entry':command+=['--','--difficulty=open2']
    if name=='test_m8x_game_entry':command+=['--','--difficulty=open3']
    run(name,command)
sys.exit(0 if all(r['passed'] for r in checks) else 1)
