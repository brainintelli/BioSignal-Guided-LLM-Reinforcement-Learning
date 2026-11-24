"""Bounded independent GPU jobs; each host receives whole paired seed blocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time
from .data import Dataset
from .features import digest
from .protocol import PROTOCOL,ARMS,cell,assignment
from .train import write
from .integrity import verify_assets,file_sha


def persist(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def verify_bundle(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Registered source/data changed: '+name)
    return hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()


def execute(args,block,arm,gpu,registration_id):
    root=args.root
    out=root/'experiment01'/f'b{block:02d}_{arm}'
    out.mkdir(parents=True,exist_ok=False)
    status=dict(block=block,arm=arm,gpu=gpu,host=socket.gethostname(),registration_sha256=registration_id,
                passed=False,complete=False,stages=[],started_utc=datetime.now(timezone.utc).isoformat())
    persist(out/'status.json',status)
    common=['--root',str(root),'--source',str(root/'data'),'--project',str(root/'project'),
            '--model-config',str(root/'model_config.json'),'--godot',str(args.godot),'--arm',arm,'--block',str(block)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    stages=[]
    if arm!='A0':stages.append(('train',['--output',str(out/'train')]))
    stages.append(('evaluate',['--output',str(out/'eval')]+([] if arm=='A0' else ['--checkpoint',str(out/'train/round_03/checkpoint')])))
    stages.append(('verify',['--output',str(out/'verification'),'--eval-dir',str(out/'eval')]+
                   ([] if arm=='A0' else ['--train-dir',str(out/'train')])))
    try:
        for stage,extra in stages:
            status['stage']=stage;persist(out/'status.json',status)
            command=[sys.executable,'-u','-m','mea_state2_r2.train',stage,*common,*extra]
            started=time.monotonic()
            with (out/(stage+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=root/'project',env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,timeout=10800)
            status['stages'].append(dict(stage=stage,exit_code=result.returncode,seconds=time.monotonic()-started))
            persist(out/'status.json',status)
            if result.returncode:raise RuntimeError(stage+' failed; no automatic retry')
        verified=json.loads((out/'verification/verified.json').read_text())
        if verified.get('passed') is not True:raise ValueError('Verification not accepted')
        evaluation=json.loads((out/'eval/evaluation.json').read_text())
        summary=dict(block=block,arm=arm,passed=True,smoke=False,host=status['host'],gpu=gpu,
                     metrics=evaluation['metrics'],evaluated=evaluation['evaluated'],registration_sha256=registration_id,
                     verification_sha256=file_sha(out/'verification/verified.json'),evaluation_sha256=file_sha(out/'eval/evaluation.json'))
        if arm!='A0':
            summary['training']=verified['training_summary']
        write(out/'cell_summary.json',summary)
        status['passed']=True
    except Exception as exc:
        status['error']=type(exc).__name__+': '+str(exc)
    status.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat())
    persist(out/'status.json',status)
    return status


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--godot',required=True,type=Path)
    p.add_argument('--blocks',required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    a=p.parse_args()
    blocks=[int(x) for x in a.blocks.split(',')];gpus=[int(x) for x in a.gpus.split(',')]
    if not blocks or len(set(blocks))!=len(blocks) or not gpus or len(set(gpus))!=len(gpus) or any(g<0 or g>7 for g in gpus):
        raise ValueError('Unique bounded blocks/GPUs required')
    for b in blocks:cell(b,'A0')
    source_id=verify_bundle(a.root)
    assets=verify_assets(a.root)
    write(a.root/'assets_verified.json',assets)
    data=Dataset(a.root/'data')
    registration=dict(protocol=PROTOCOL,source_manifest_sha256=source_id,external_assets_sha256=assets['external_assets_sha256'],data=data.summary(),
        schedules=[cell(b,arm) for b in range(8) for arm in ('A0',*ARMS)],
        assignments={str(b):{arm:{split:assignment(b,arm,split) for split in ('train','eval')} for arm in ARMS} for b in range(8)})
    write(a.root/'registration.json',registration)
    launch=dict(host=socket.gethostname(),blocks=blocks,gpus=gpus,registration_sha256=digest(registration),
                automatic_restart=False,planned_cells=len(blocks)*5,complete=False,passed=False)
    statuspath=a.root/'campaign_status.json';persist(statuspath,launch)
    jobs=queue.Queue()
    for b in blocks:jobs.put((b,'A0'))
    for arm in ARMS:
        for b in blocks:jobs.put((b,arm))
    def worker(gpu):
        results=[]
        while True:
            try:block,arm=jobs.get_nowait()
            except queue.Empty:break
            results.append(execute(a,block,arm,gpu,digest(registration)))
        return results
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        outcomes=[row for group in pool.map(worker,gpus) for row in group]
    if verify_assets(a.root)!=assets or verify_bundle(a.root)!=source_id:
        raise ValueError('Assets changed during campaign')
    launch.update(complete=True,passed=len(outcomes)==len(blocks)*5 and all(r['passed'] for r in outcomes),
                  accepted_cells=sum(r['passed'] for r in outcomes),outcomes=outcomes,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    persist(statuspath,launch)
    raise SystemExit(0 if launch['passed'] else 1)


if __name__=='__main__':main()
