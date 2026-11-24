"""One explicit launch of the fixed session, retaining exit status and logs."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path

BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
PYTHON='/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python'

if sys.argv[1:]==['--worker']:
    env=dict(os.environ,HDF5_PLUGIN_PATH='/pub/soft/MaxLab/so',PYTHONUNBUFFERED='1')
    with (BASE/'run.log').open('x') as log:
        process=subprocess.Popen([PYTHON,str(BASE/'mea_next_recording/run.py'),'run',
                     '--binding',str(BASE/'binding.json'),'--output',str(BASE/'recording')],
                     stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        job={'worker_pid':os.getpid(),'child_pid':process.pid,
             'started_utc':datetime.now(timezone.utc).isoformat(),'automatic_restart':False}
        (BASE/'job.json').write_text(json.dumps(job,indent=2))
        job['exit_code']=process.wait();job['finished_utc']=datetime.now(timezone.utc).isoformat()
        (BASE/'job.json').write_text(json.dumps(job,indent=2))
    raise SystemExit(job['exit_code'])
if sys.argv[1:]!=['--launch']:raise SystemExit('Explicit --launch required')
route=Path('/home/researcher/configs/261117/13h59m04s.cfg').read_bytes()
assert hashlib.sha256(route).hexdigest()=='4129baeb772b604ba265a4e3a4feb10d5fe0a04a70c615260e286514646d51e1'
with (BASE/'launch_claim.json').open('x') as f:
    json.dump({'started_utc':datetime.now(timezone.utc).isoformat(),'single_launch':True},f);f.flush();os.fsync(f.fileno())
(BASE/'route_frozen.cfg').write_bytes(route)
binding={'protocol':'photo_full_v1','chip_id':'28080','session_started_at':'2026-11-17T14:06:00+08:00',
         'acquisition_host':'192.0.2.101','route_cfg':str(BASE/'route_frozen.cfg'),
         'alpha':1,'r2_mode':'record_only','realtime_training':False,'offline_training':False}
(BASE/'binding.json').write_text(json.dumps(binding,indent=2))
with (BASE/'worker.log').open('x') as log:
    process=subprocess.Popen([PYTHON,str(BASE/'launch_session.py'),'--worker'],stdin=subprocess.DEVNULL,
               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'worker_pid':process.pid,'base':str(BASE),'deadline_beijing':'2026-11-17T16:06:00+08:00'}))
