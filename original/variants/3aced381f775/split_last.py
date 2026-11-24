"""Spread the final seed across idle GPUs, keeping every map and its original random state."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text())
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def main(seed):
    full=read(HERE/'design.json')
    assert seed==full['seeds'][-1]
    folders=[HERE/'shards'/str(i) for i in range(3)]
    for i,folder in enumerate(folders):
        folder.mkdir(parents=True,exist_ok=True)
        spec=full|{'worlds':full['worlds'][16*i:16*(i+1)]}
        p=folder/'design.json'
        if p.exists():assert read(p)==spec
        else:p.write_text(json.dumps(spec,ensure_ascii=False))
        for name in ('audit-game','plain-game'):
            link=folder/name
            if not link.exists():link.symlink_to(HERE/name,target_is_directory=True)
    todo=folders.copy()
    initial_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    running={}
    while todo or running:
        for g,(p,log) in list(running.items()):
            if p.poll() is None:continue
            log.close()
            if p.returncode:raise RuntimeError((p.args,p.returncode))
            del running[g]
        if todo:
            entries=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
            gpus=sorted((tuple(map(int,line.split(','))) for line in entries),key=lambda x:(x[0]!=initial_gpu,-x[1]))
            for gpu,free in gpus:
                if not todo or gpu in running or free<8800:continue
                task=todo.pop(0)
                log=(task/'worker.log').open('a')
                cmd=[sys.executable,'-u',HERE/'run.py',str(seed),str(task)]
                proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                running[gpu]=(proc,log)
                print('map block',task.name,'on',gpu,flush=True)
        time.sleep(2)
    out=HERE/'runs'/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    provenance={}
    for mode in full['modes']:
        games=[];weights=None
        for folder in folders:
            piece=read(folder/'design.json')
            assert piece==full|{'worlds':piece['worlds']}
            path=folder/'runs'/str(seed)/(mode+'.json')
            raw=read(path)
            assert raw['study_sha']==canonical(piece) and raw['seed']==seed and raw['mode']==mode
            assert piece['worlds']==[g['world'] for g in raw['games']]
            assert weights is None or weights==raw['frozen']
            weights=raw['frozen'];games.extend(raw['games'])
            provenance[path.relative_to(HERE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert [g['world'] for g in games]==full['worlds']
        doc=dict(seed=seed,mode=mode,study_sha=canonical(full),frozen=weights,games=games)
        tmp=out/(mode+'.partial')
        tmp.write_text(json.dumps(doc,separators=(',',':')))
        tmp.replace(out/(mode+'.json'))
    (HERE/'shards/constituents.json').write_text(json.dumps(provenance,indent=2))
