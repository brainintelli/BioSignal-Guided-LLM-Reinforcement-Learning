"""Verify frozen policies and run ordered, masked and time-permuted inputs on a second recording."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
import torch
from agent.m9_env import GodotEnv
from agent.m9_policy import load
from mea_state2_r2.features import digest
from play60 import play,ARCHIVE
from resources import GODOT,PROJECT
from tools.mea_shuffle_train import state
from train import FixedContextPolicy,read

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(seed,condition):
    spec=read(HERE/'design.json');parent=read(PARENT/'design.json')
    assert sha(HERE/'evaluate.py')==spec['runner_sha'] and sha(PARENT/'play60.py')==spec['previous_play_sha']==parent['eval_code_sha']
    assert digest(parent)==spec['previous_study_sha'] and sha(PARENT/'train.py')==parent['training_code_sha']
    assert sha(PARENT/'training-audit.json')==spec['training_audit_sha'] and seed in spec['training_seeds']
    training_arm,input_arm=spec['conditions'][condition]
    model=PARENT/'models'/str(seed)/training_arm
    reg=read(model/'registration.json');up=read(model/'update.json');manifest=read(model/'checkpoint/manifest.json')
    expected=spec['states'][f'{seed}/{training_arm}']; assert expected==up['after'] and manifest['actor_sha256']==expected['actor']
    assert reg['study_digest']==digest(parent) and reg['arm']==training_arm and reg['training_seed']==seed
    assert all(sha(model/'checkpoint'/nm)==h for nm,h in manifest['files'].items())
    ctx=read(PARENT/'context.json');assert digest(ctx)==parent['context_sha'] and ctx['normalization']==spec['normalization'] and ctx['roi_sha']==spec['roi_sha']
    torch.manual_seed(6100000)
    cfg=reg['config']|read(PARENT/'model_config.json')|{'rng_seed':6100000}
    policy=FixedContextPolicy(SimpleNamespace(**cfg),training_arm,ctx)
    assert state(policy)==reg['initial'];load(policy,model/'checkpoint');assert state(policy)==expected
    policy.context=ctx|dict(source_sha256=spec['source_sha'],state2_source='archived_prestimulation_quiet_recording')
    policy.arm='A' if input_arm=='masked' else 'C'
    out=HERE/'games'/str(seed)/condition;out.mkdir(parents=True,exist_ok=True);pin=digest(spec)
    with torch.inference_mode(),GodotEnv(str(GODOT),str(ARCHIVE/'audit-game'),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(PROJECT),map_mode='infinite_freeze',navigation_steps=4) as plain:
        for w in spec['worlds']:
            file=out/f'{w}.json'
            if file.exists():
                v=read(file)
                assert v['spec']==pin and v['frozen']==expected and v['condition']==condition and v['game']['world']==w
                continue
            game=play(policy,env,plain,spec,w,input_arm)
            part=file.with_suffix('.tmp')
            part.write_text(json.dumps(dict(spec=pin,seed=seed,condition=condition,frozen=expected,game=game),separators=(',',':')))
            part.replace(file)
    assert state(policy)==expected and len(list(out.glob('[0-9]*.json')))==len(spec['worlds'])
    (out/'frozen.json').write_text(json.dumps(dict(spec=pin,seed=seed,condition=condition,state=expected,worlds=len(spec['worlds']))))
    print('complete',seed,condition,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
