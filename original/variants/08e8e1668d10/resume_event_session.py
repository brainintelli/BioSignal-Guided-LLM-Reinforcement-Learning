"""Explicit reviewed continuation after the unexecuted malformed Event setup."""
from datetime import datetime,timezone
import json,os,subprocess,sys,time
from pathlib import Path
BASE=Path('/home/researcher/mea_full_28080_20261117_1406')
PYTHON='/home/researcher/chenyuzhe/2026-9-8/BrainAI/.venv/bin/python'
sys.path.insert(0,str(BASE/'corrected/mea_next_recording'))
from protocol import validate_binding,session_key,atomic_json
from recovery import VerifiedEventSetupResume,verify_event_setup_resume

if sys.argv[1:]==['--run']:
    from device import MaxLabBackend
    from run import deadline_signals
    b=json.loads((BASE/'binding.json').read_text());start=validate_binding(b,time.time())
    backend=MaxLabBackend(b,BASE/'recording_resume2');journal=Path.home()/'.local/state/mea_photo_full/sessions'
    with backend.sdk_context(),deadline_signals(start+7200-time.time()-120,backend):
        report=VerifiedEventSetupResume(b,backend,time,BASE/'recording_resume2',journal).run()
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='completed' else 1)
if sys.argv[1:]==['--worker']:
    env=dict(os.environ,HDF5_PLUGIN_PATH='/pub/soft/MaxLab/so',PYTHONUNBUFFERED='1')
    with (BASE/'event_resume_run.log').open('x') as log:
        p=subprocess.Popen([PYTHON,str(BASE/'resume_event_session.py'),'--run'],stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,env=env)
        job=dict(worker_pid=os.getpid(),child_pid=p.pid,started_utc=datetime.now(timezone.utc).isoformat(),automatic_restart=False)
        atomic_json(BASE/'event_resume_job.json',job)
        job['exit_code']=p.wait();job['finished_utc']=datetime.now(timezone.utc).isoformat();atomic_json(BASE/'event_resume_job.json',job)
    raise SystemExit(job['exit_code'])
if sys.argv[1:]!=['--launch']:raise SystemExit('Explicit --launch required')
b=json.loads((BASE/'binding.json').read_text());start=validate_binding(b,time.time())
old=json.loads((BASE/'resume_job.json').read_text());assert old['exit_code']==1
assert not Path('/proc/'+str(old['child_pid'])).exists()
key=session_key(b,start);journal=Path.home()/'.local/state/mea_photo_full/sessions'
proof=verify_event_setup_resume(BASE/'recording_resume1',b,journal/(key+'.stimulus.json'))
assert json.loads((BASE/'sequence_upload_only/verification.json').read_text())['status']=='passed'
atomic_json(BASE/'reviewed_event_resume_launch.json',dict(proof=proof,deadline='2026-11-17T16:06:00+08:00'),exclusive=True)
with (BASE/'event_resume_worker.log').open('x') as log:
    p=subprocess.Popen([PYTHON,str(BASE/'resume_event_session.py'),'--worker'],stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(worker_pid=p.pid,verified_event_setup_recovery=True,original_stimulus_budget_unchanged=True)))
