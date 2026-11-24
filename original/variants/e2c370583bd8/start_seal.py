import json,subprocess,sys
from pathlib import Path
r=Path('/home/researcher/mea-stim-offline-20261117')
if (r/'seal_job.json').exists():raise RuntimeError('Seal already started')
if json.loads((r/'job.json').read_text()).get('accepted_complete') is not True:raise RuntimeError('Experiment incomplete')
with (r/'seal.log').open('x') as log:
    p=subprocess.Popen([sys.executable,str(r/'seal_delivery.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (r/'seal_job.json').open('x') as f:json.dump({'pid':p.pid,'automatic_restart':False},f)
print(p.pid)
