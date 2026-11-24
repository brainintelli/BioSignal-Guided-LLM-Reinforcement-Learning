"""Recreate a frozen policy's actions, then verify the 192 native ticks in another Godot process."""
import json
import sys
from pathlib import Path

from clock_replay import replay_seed, require, sha
from mea_online.m9_runtime import M9Runtime, digest
from train_long_recorded import decisions, read, windows


def verify(root, jobid):
    import torch
    plan=read(root/'protocol.json');job=plan['models'][jobid]
    cp=Path(job['checkpoint']);reg=read(cp.parent/'registration.json');cfg=reg['config']
    gs=plan['maps'][0];stored=read(root/'runs'/jobid/f'{gs}.json')
    require(sha(cp/'manifest.json')==job['manifest'] and digest(reg)==job['registration'], 'Checkpoint changed')
    order=[];prime=None
    if cfg['arm']!='pure':
        order=windows(read(root/'replay.json'),cfg['arm'],False)
        prime=read(root/'preceding-quiet.json')['prime']
    torch.manual_seed(cfg['seed'])
    rt=M9Runtime(cfg,root/'checks'/jobid,frozen_evaluation=True)
    trace=[]
    try:
        rt.core.load_checkpoint(cp,digest(reg))
        elapsed=0
        for row,step,_,_ in decisions(rt,cfg,order,prime,gs,192,False):
            trace.append(step);elapsed+=row['duration']
            if elapsed>=192:break
        require(rt.state()==job['state'],'Changed weights')
    finally:
        rt.close()
    _,_,out=replay_seed((cfg['arm'],cfg['seed'],{str(gs):trace},cfg['project'],cfg['godot'],192))
    obs=out[str(gs)]
    require(stored['ticks']==obs['series'] and stored['first_obs_hash']==trace[0]['obs_hash']
            and stored['terminal_tick']==obs['terminal_tick']
            and [x[1] for x in stored['actions']]==[s['action']['verb'] for s in trace],
            'Independent game replay differs')
    return dict(model=jobid,map_seed=gs,matched_ticks=192,decisions=len(trace),manifest=job['manifest'],
                saved_game_sha=sha(root/'runs'/jobid/f'{gs}.json'))


if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1]),sys.argv[2])))
