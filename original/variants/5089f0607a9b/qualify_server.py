"""Eight independent short model updates/reloads on a newly provisioned host."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

root=Path(sys.argv[1]).resolve()
runtime=Path('/home/researcher/mea-r2-22554/runtime_20261116_01/assets')
assert json.loads((root/'preflight.json').read_text())['passed'] is True
out=root/'qualification01';out.mkdir(exist_ok=False)
base_env=dict(os.environ,**json.loads((root/'environment.json').read_text()))
common=['--root',str(root),'--source',str(root/'data'),'--project',str(root/'project'),
    '--model-config',str(root/'model_config.json'),'--godot',str(runtime/'godot/Godot_v4.7.2-stable_linux.x86_64'),
    '--arm','C','--smoke']


def run(gpu):
    cell=out/f'gpu{gpu}';cell.mkdir();stages=[]
    env=dict(base_env,CUDA_VISIBLE_DEVICES=str(gpu))
    record=dict(gpu=gpu,block=gpu,technical_only=True,efficacy_included=False,passed=False)
    try:
        for mode,args in [('train',['--output',str(cell/'train')]),
                          ('verify',['--output',str(cell/'verification'),'--train-dir',str(cell/'train')])]:
            command=[str(runtime/'python/bin/python'),'-u','-m','mea_state2_r2.train',mode,*common,'--block',str(gpu),*args]
            started=time.monotonic()
            with (cell/(mode+'.log')).open('x') as log:
                p=subprocess.run(command,cwd=root/'project',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            stages.append(dict(mode=mode,exit_code=p.returncode,seconds=time.monotonic()-started))
            if p.returncode:raise RuntimeError(mode+' failed; no automatic retry')
        report=json.loads((cell/'verification/verified.json').read_text())
        assert report['passed'] is True and report['smoke'] is True and report['training_decisions']==16
        record.update(passed=True,verified=report)
    except Exception as exc:record['error']=type(exc).__name__+': '+str(exc)
    record['stages']=stages
    (cell/'status.json').write_text(json.dumps(record,indent=2));return record


with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(run,range(8)))
summary=dict(host=socket.gethostname(),passed=all(r['passed'] for r in results),technical_only=True,
    efficacy_included=False,hardware_commands_sent=0,results=results)
(out/'result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'eight_gpu_qualification_passed':summary['passed'],'accepted':sum(r['passed'] for r in results)}),flush=True)
raise SystemExit(0 if summary['passed'] else 1)
