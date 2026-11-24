import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1])
host=sys.argv[2]
code=Path(__file__).resolve().parent
plan=json.loads((root/'paired-protocol.json').read_bytes())
env=dict(os.environ)
env.update(json.loads((Path.home()/'mea-state2-r2-20261118/run02/environment.json').read_bytes()))
env.update(PYTHONPATH=str(code)+os.pathsep+env.get('PYTHONPATH',''),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
py=str(Path.home()/'mea-r2-22554/runtime_20261116_01/assets/python/bin/python')
jobs=[]
for i,seed in enumerate(plan['training_seeds']):
    arm='ordered' if (host=='16')==(i%2==0) else 'shuffled'
    jobs.append((arm,seed))

def launch(mode,job,gpu=None):
    arm,seed=job
    out=root/'runs'/f'{arm}-{seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/('trained.json' if mode=='train' else 'done.json')).is_file():return True
    settings=dict(env)
    if gpu is not None:settings['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (out/(mode+'.log')).open('ab') as f:
        rc=subprocess.run([py,'-u',str(code/'study.py'),str(root),mode,arm,str(seed)],env=settings,
                          stdout=f,stderr=f,stdin=subprocess.DEVNULL).returncode
    flag=out/(mode+'.error')
    if rc:flag.write_text(str(rc))
    else:flag.unlink(missing_ok=True)
    return not rc

for j in jobs:
    old_root=str(root.with_name('22554-hwf-1119-paired')).encode()
    def previous_fit_running():
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=proc.read_bytes().split(b'\0')
            except (PermissionError,FileNotFoundError,ProcessLookupError):continue
            if old_root in args and b'train' in args and any(a.endswith(b'/study.py') for a in args):return True
        return False
    while previous_fit_running():time.sleep(8)
    if not launch('train',j):sys.exit(1)
jobs.extend(('pure',s) for s in plan['training_seeds'] if plan['pure'][str(s)]['host']==host)
def lane(gpu):
    return all(launch('eval',j,gpu) for j in jobs[gpu::8])
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    codes=list(pool.map(lane,range(8)))
sys.exit(0 if all(codes) else 2)
