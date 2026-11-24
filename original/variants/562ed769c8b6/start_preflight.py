import json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
if (r/'preflight_launch.json').exists():raise FileExistsError('No automatic retry')
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
with (r/'preflight_worker.log').open('x') as log:
    p=subprocess.Popen([sys.executable,str(r/'preflight_worker.py')],cwd=r,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'preflight_launch.json').write_text(json.dumps({'pid':p.pid}));print(json.dumps({'pid':p.pid}),flush=True)
