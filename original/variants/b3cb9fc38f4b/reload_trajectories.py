"""Independent frozen checkpoint reload on registered map position zero in each scene."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STRESS = RUN/'stress'
sys.path.insert(0,str(RUN))
from resources import environment, PYTHON


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model(seed, arm):
    import torch
    from types import SimpleNamespace
    sys.path.insert(0,str(STRESS))
    from evaluate import GodotEnv, GODOT, load, digest, state, FixedContextPolicy, play, ResourceTrace, append_resources
    spec=read(STRESS/'design.json'); model=RUN/'models'/str(seed)/arm
    world=spec['worlds'][0]; ctx=read(RUN/'context.json'); reg=read(model/'registration.json')
    assert reg['study_digest']==digest(read(RUN/'design.json')) and sha(RUN/'play60.py')==spec['play_sha']
    assert digest(ctx['series'])==reg['windows_digest']
    warm_cfg=reg['config']|read(RUN/'model_config.json')|{'rng_seed':6100000}
    torch.manual_seed(6100000)
    policy=FixedContextPolicy(SimpleNamespace(**warm_cfg),arm,ctx)
    assert state(policy)==reg['initial']
    load(policy,model/'checkpoint')
    frozen=spec['states'][f'{seed}/{arm}']
    assert state(policy)==frozen
    policy.context=ctx|dict(source_sha256=spec['source_sha'],state2_source='heldout_crossover_blocks_5_6')
    checked=[]
    with torch.inference_mode():
        for scene,reserves in spec['scenes'].items():
            game_spec=spec|dict(starting_scene='stress_'+scene,starting_reserves=reserves)
            actual=read(STRESS/'games'/str(seed)/scene/arm/f'{world}.json')
            assert actual['frozen']==frozen and actual['spec']==digest(spec)
            with GodotEnv(str(GODOT),str(HERE/'native'/scene/'audit-game'),map_mode='infinite_freeze',navigation_steps=4) as obs, GodotEnv(str(GODOT),str(STRESS/'scenes'/scene/'plain'),map_mode='infinite_freeze',navigation_steps=4) as plain:
                trace=ResourceTrace(plain)
                replayed=play(policy,obs,trace,game_spec,world,arm);append_resources(replayed,trace)
            original=actual['game']
            for key in ['world','mode','decisions','end_tick','survived','new_cells','first_success_tick','restricted_wait','health','thirst','hunger','cut']:
                assert replayed[key]==original[key],(seed,arm,scene,key)
            fields=['tick','end_tick','window','events','features','action','transition_sha','cut','new_cells_at_end','r2_logged','r2_contribution','resource_event']
            def trajectory(game):return [{k:row[k] for k in fields} for row in game['rows']]
            assert trajectory(replayed)==trajectory(original),(seed,arm,scene,'decisions')
            checked.append(dict(scene=scene,world=world,decisions=len(original['rows']),trajectory_sha256=digest(trajectory(original))))
    assert state(policy)==frozen
    return dict(seed=seed,arm=arm,checkpoint_state=frozen,checked=checked)


def check_gpu(gpu, pairs):
    env=environment() | {'CUDA_VISIBLE_DEVICES':str(gpu),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    for seed,arm in pairs:
        jobdir=HERE/'reloads'; dest=jobdir/f'{seed}-{arm}.json'
        if dest.exists():
            row=read(dest)
            if row['seed']==seed and row['arm']==arm and row['verifier_sha256']==sha(Path(__file__)):continue
            raise ValueError(f'Unexpected previous receipt {dest}')
        log=jobdir/f'{seed}-{arm}.log'
        with log.open('w') as out:
            p=subprocess.run([str(PYTHON),str(Path(__file__)),str(seed),arm],env=env,stdout=out,stderr=subprocess.STDOUT)
        if p.returncode: raise RuntimeError(f'{seed} {arm}: {log.read_text(errors="replace")[-1150:]}')
        print('verified',seed,arm,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int,nargs='?');parser.add_argument('arm',nargs='?');args=parser.parse_args()
    (HERE/'reloads').mkdir(exist_ok=True)
    if args.seed is not None:
        row=verify_model(args.seed,args.arm);row['verifier_sha256']=sha(Path(__file__))
        f=HERE/'reloads'/f'{args.seed}-{args.arm}.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(f);return
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus=[int(v.split(',')[0]) for v in q.splitlines() if int(v.split(',')[1])>=12500]
    if not gpus:raise RuntimeError('No GPU has sufficient free memory')
    spec=read(STRESS/'design.json');pairs=[(seed,arm) for seed in spec['training_seeds'] for arm in spec['arms']]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs=[pool.submit(check_gpu,gpu,pairs[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for j in as_completed(jobs):j.result()
    rows=[read(HERE/'reloads'/f'{s}-{a}.json') for s,a in pairs]
    assert all(len(r['checked'])==3 and r['verifier_sha256']==sha(Path(__file__)) for r in rows)
    out=dict(selection='outcome-independent post-run QC, first registered world for every checkpoint and scene',world=spec['worlds'][0],models=len(rows),episodes=sum(len(r['checked']) for r in rows),decisions=sum(x['decisions'] for r in rows for x in r['checked']),action_mismatches=0,after_matches=96,verifier_sha256=sha(Path(__file__)),records=rows)
    assert out['episodes']==288 and out['models']==96
    f=HERE/'policy-reload.json';tmp=f.with_suffix('.json.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(f);print({k:v for k,v in out.items() if k!='records'}, flush=True)

if __name__=='__main__':main()
