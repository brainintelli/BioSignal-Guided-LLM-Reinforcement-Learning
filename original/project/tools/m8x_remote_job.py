"""Detached, identity-checked lifecycle for one authorized M8B training run."""
from __future__ import annotations
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
from agent.m8x_registration import validate_config

ROOT=Path.home()/'meadowmind-gpu'
JOBS=ROOT/'m8b-evidence/jobs'

def identity(pid):
    proc=Path(f'/proc/{pid}');stat=(proc/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    if fields[0]=='Z':raise ProcessLookupError(pid)
    return {'pid':pid,'start_ticks':fields[19],'executable':str((proc/'exe').resolve())}

def owned(saved):
    try:current=identity(saved['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    if any(saved[k]!=v for k,v in current.items()):raise RuntimeError('Identity differs; process left untouched')
    return True

def write(path,value):
    temp=path.with_suffix('.writing');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def stop_child(saved,process):
    if not owned(saved):return
    if os.getpgid(saved['pid'])!=saved['pid']:raise RuntimeError('Child process group differs')
    os.killpg(saved['pid'],signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if owned(saved):os.killpg(saved['pid'],signal.SIGKILL)
        process.wait(timeout=5)

def worker(directory):
    launch=json.loads((directory/'launch.json').read_text());config=Path(launch['config'])
    if hashlib.sha256(config.read_bytes()).hexdigest()!=launch['config_sha256']:raise ValueError('Config changed after launch')
    stopping=False
    def terminate(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    began=time.monotonic();reason='completed';process=None;saved=None
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH=launch['pythonpath'])
    try:
        with (directory/'training.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(launch['argv'],cwd=ROOT/'m8b-project',env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=log,start_new_session=True)
        time.sleep(.05)
        try:
            saved=identity(process.pid);saved.update({'command':launch['argv']});write(directory/'child.json',saved)
        except (FileNotFoundError,ProcessLookupError):
            if process.poll() is None:raise
            write(directory/'child.json',{'pid':process.pid,'already_exited':True,'command':launch['argv']})
        while process.poll() is None:
            if stopping or time.monotonic()-began>=launch['hard_wall_seconds']:
                reason='requested_stop' if stopping else 'hard_wall_limit';stop_child(saved,process);break
            time.sleep(1)
        result={'exit_code':process.returncode,'reason':reason,'seconds':time.monotonic()-began,
                'finished_utc':datetime.now(timezone.utc).isoformat()}
        write(directory/'finished.json',result)
    except BaseException as error:
        if process is not None and saved is not None and process.poll() is None:stop_child(saved,process)
        write(directory/'finished.json',{'exit_code':process.poll() if process else None,'reason':'manager_error',
              'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-began})
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','status','stop','_run'])
    p.add_argument('--name',required=True);p.add_argument('--config');p.add_argument('--hard-wall-seconds',type=int,default=172800)
    a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',a.name):raise ValueError('Invalid job name')
    directory=JOBS/a.name
    if a.action=='_run':worker(directory);return
    record=directory/'manager.json';saved=json.loads(record.read_text()) if record.exists() else None
    live=owned(saved) if saved else False
    if a.action=='status':
        result={'running':live,'manager':saved,'directory':str(directory)}
        for file in ['child.json','finished.json','launch.json']:
            if (directory/file).exists():result[file.removesuffix('.json')]=json.loads((directory/file).read_text())
        print(json.dumps(result,indent=2));return
    if a.action=='stop':
        if live:
            os.kill(saved['pid'],signal.SIGTERM)
            for _ in range(200):
                if not owned(saved):break
                time.sleep(.1)
            else:raise RuntimeError('Manager did not stop; no unverified process was killed')
        print('OWNED_M8_JOB_STOPPED');return
    if live or directory.exists():raise RuntimeError('Job name exists; choose a new name to preserve evidence')
    if not a.config or not 1<=a.hard_wall_seconds<=172800:raise ValueError('Config and 1..172800 second wall limit required')
    config=Path(a.config).resolve()
    if not config.is_relative_to((ROOT/'m8b-evidence').resolve()):raise ValueError('Config outside m8b-evidence')
    data=json.loads(config.read_text());project=Path(data['project']).resolve();output=Path(data['output']).resolve()
    model=Path(data['model']).resolve();godot=Path(data['godot']).resolve()
    if project!=(ROOT/'m8b-project').resolve():raise ValueError('Unexpected project path')
    if not output.is_relative_to((ROOT/'m8b-runs').resolve()) or output.exists():raise ValueError('Output must be new within m8b-runs')
    if not model.is_relative_to((ROOT/'models').resolve()) or not model.is_dir():raise ValueError('Model outside project models')
    if not godot.is_relative_to((ROOT/'godot').resolve()) or not godot.is_file():raise ValueError('Unknown Godot runtime')
    validate_config(project,data)
    overlay=Path(data['kernel_overlay']).resolve()
    if not overlay.is_relative_to((ROOT/'m8b-runtime-candidates').resolve()) or not overlay.is_dir():
        raise ValueError('Kernel overlay must be the tested isolated runtime')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU has compute work; left untouched: '+active)
    directory.mkdir(parents=True)
    frozen=directory/'config.json';frozen.write_bytes(config.read_bytes())
    argv=[str(ROOT/'m6-venv/bin/python'),'-u','-m','agent.m8x_train','--config',str(frozen)]
    launch={'argv':argv,'config':str(frozen),'config_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
            'pythonpath':str(overlay)+':'+str(project),
            'output':str(output),'hard_wall_seconds':a.hard_wall_seconds,'created_utc':datetime.now(timezone.utc).isoformat()}
    write(directory/'launch.json',launch)
    with (directory/'manager.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([str(ROOT/'m6-venv/bin/python'),str(Path(__file__).resolve()),'_run','--name',a.name],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    time.sleep(.15);saved=identity(process.pid);write(record,saved);print(json.dumps({'manager':saved,**launch},indent=2))

if __name__=='__main__':main()
