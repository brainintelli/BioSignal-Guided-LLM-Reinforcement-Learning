"""Detached exploration frozen assessment lifecycle; never starts by import."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import m8x_remote_job as lifecycle
from agent.m8x_eval import authorize_test

ROOT=lifecycle.ROOT
JOBS=ROOT/'m8b-evidence/assessments'
MODULES={'exploration-stochastic':('m6-venv','agent.m8x_eval')}


def checked_path(value, root, *, new=False):
    path=Path(value).resolve()
    if not path.is_relative_to(root.resolve()) or path==root.resolve():
        raise ValueError('Path outside assessment scope: '+str(path))
    if new and path.exists():raise ValueError('Evidence path already exists: '+str(path))
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','status','stop','_run'])
    p.add_argument('--name',required=True);p.add_argument('--kind',choices=MODULES)
    p.add_argument('--request');p.add_argument('--hard-wall-seconds',type=int,default=172800)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',a.name):raise ValueError('Invalid name')
    directory=JOBS/a.name
    if a.action=='_run':lifecycle.worker(directory);return
    record=directory/'manager.json'
    saved=json.loads(record.read_text()) if record.exists() else None
    live=lifecycle.owned(saved) if saved else False
    if a.action=='status':
        result={'running':live,'manager':saved,'directory':str(directory)}
        for name in ['launch','child','finished']:
            file=directory/(name+'.json')
            if file.exists():result[name]=json.loads(file.read_text())
        print(json.dumps(result,indent=2));return
    if a.action=='stop':
        if live:
            os.kill(saved['pid'],signal.SIGTERM)
            for _ in range(200):
                if not lifecycle.owned(saved):break
                time.sleep(.1)
            else:raise RuntimeError('Manager did not stop; other processes untouched')
        print('OWNED_ASSESSMENT_STOPPED');return
    if live or directory.exists():raise ValueError('Assessment name exists; preserve evidence')
    if not a.kind or not a.request or not 1<=a.hard_wall_seconds<=172800:
        raise ValueError('Kind, request and bounded wall time required')
    request=checked_path(a.request,ROOT/'m8b-evidence')
    data=json.loads(request.read_text())
    output=checked_path(data['output'],ROOT/'m8b-runs',new=True)
    checked_path(data['run'],ROOT/'m8b-runs')
    if data['mode'] != 'test':
        raise ValueError('Exploration assessment supports only the independently authorized test')
    config, *_ = authorize_test(data)
    project=Path(config['project']).resolve()
    if project != (ROOT/'m8b-project').resolve():raise ValueError('Unexpected registered project')
    overlay=checked_path(config['kernel_overlay'],ROOT/'m8b-runtime-candidates')
    if not overlay.is_dir():raise ValueError('Tested isolated kernel runtime is missing')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU busy; untouched: '+active)
    venv,module=MODULES[a.kind]
    directory.mkdir(parents=True)
    frozen=directory/'request.json';frozen.write_bytes(request.read_bytes())
    argv=[str(ROOT/venv/'bin/python'),'-u','-m',module,'--request',str(frozen)]
    launch={'argv':argv,'config':str(frozen),'config_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
            'pythonpath':str(overlay)+':'+str(project),
            'output':str(output),'kind':a.kind,'hard_wall_seconds':a.hard_wall_seconds,
            'created_utc':datetime.now(timezone.utc).isoformat()}
    lifecycle.write(directory/'launch.json',launch)
    with (directory/'manager.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([str(ROOT/'m6-venv/bin/python'),str(Path(__file__).resolve()),'_run','--name',a.name],
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    time.sleep(.15);saved=lifecycle.identity(process.pid);lifecycle.write(record,saved)
    print(json.dumps({'manager':saved,**launch},indent=2))

if __name__=='__main__':main()
