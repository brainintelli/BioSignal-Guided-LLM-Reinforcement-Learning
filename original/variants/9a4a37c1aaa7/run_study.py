"""Allocate available GPUs, preserve failed attempts, and audit every completed model."""
import json
import os
from pathlib import Path
import subprocess
import time
from resources import ROOT, PYTHON, GODOT, PROJECT, environment

def gpu_free():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(k):int(v) for k,v in (row.split(',') for row in rows.splitlines())}

def run():
    spec=json.loads((ROOT/'design.json').read_text())
    pairs=[(s,a) for s in spec['training_seeds'] for a in spec['arms']]
    pending=[p for p in pairs if not (ROOT/'models'/str(p[0])/p[1]/'result.json').exists()]
    busy={}
    audit={}
    avoid={}
    while pending or busy or audit or any(not (ROOT/'models'/str(s)/a/'replay-audit.json').is_file() for s,a in pairs):
        for gpu,(proc,pair,log) in list(busy.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del busy[gpu]
            s,a=pair;folder=ROOT/'models'/str(s)/a
            if rc:
                logpath=folder.with_suffix('.log')
                if 'OutOfMemoryError' not in logpath.read_text(errors='replace') or len(avoid.get(pair,set()))>=5:raise RuntimeError((s,a,rc))
                avoid.setdefault(pair,set()).add(gpu)
                archive=ROOT/'failed-attempts'/f'{s}-{a}-gpu{gpu}'
                archive.mkdir(parents=True,exist_ok=False)
                logpath.rename(archive/'run.log');folder.rename(archive/'partial')
                pending.append(pair);print('resource retry',s,a,gpu,flush=True)
            else:print('trained',s,a,gpu,flush=True)
        for pair,(proc,log) in list(audit.items()):
            rc=proc.poll()
            if rc is not None:
                log.close();del audit[pair]
                if rc:raise RuntimeError(('audit',pair,rc))
                print('game replay verified',*pair,flush=True)
        if pending:
            for gpu,mb in sorted(gpu_free().items(),key=lambda kv:-kv[1]):
                if gpu in busy or mb<16400:continue
                eligible=next((p for p in pending if gpu not in avoid.get(p,set())),None)
                if eligible is None:continue
                s,a=eligible
                p=ROOT/'models'/str(s);p.mkdir(parents=True,exist_ok=True)
                log=(p/(a+'.log')).open('x')
                cmd=[PYTHON,'-u',ROOT/'train.py','--root',ROOT,'--project',PROJECT,'--godot',GODOT,'--model-config',ROOT/'model_config.json','--seed',str(s),'--arm',a]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(environment(),CUDA_VISIBLE_DEVICES=str(gpu)))
                busy[gpu]=(proc,eligible,log);pending.remove(eligible)
                print('start',s,a,gpu,mb,flush=True)
        for s,a in pairs:
            p=ROOT/'models'/str(s)/a
            if len(audit)>=3:break
            if (s,a) in audit or not (p/'result.json').is_file() or (p/'replay-audit.json').exists():continue
            log=(p/'audit.log').open('w')
            cmd=[PYTHON,ROOT/'verify_game.py','--root',ROOT,'--executable',GODOT,'--project',PROJECT,'--seed',str(s),'--arm',a]
            audit[(s,a)]=(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment()),log)
        time.sleep(4)
    print('complete',len(pairs),flush=True)

if __name__=='__main__':run()
