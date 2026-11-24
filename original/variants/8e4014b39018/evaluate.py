"""Replay frozen PPO checkpoints in paired worlds; log observed drinks and reward accounting."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
ARCHIVE=Path.home()/'mea-22021-20261118-mea-stim-ablation/archive-frozen-60tick'
sys.path[:0]=[str(TRAIN),str(ARCHIVE)]
from fixed_context_ppo import FixedContextPolicy,ReplayRewardMix
from resources import GODOT
from freeze import play
import torch
from agent.m9_env import GodotEnv
from agent.m9_policy import load
from livepair.predictability import outcome
from mea_state2_r2.features import digest
from mea_state2_r2.protocol import GAMMA
from tools.mea_shuffle_train import state

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class ObservedGame:
    def __init__(self,env,horizon):
        self.env=env
        self.horizon=horizon
        self.receipts=[]
    def reset(self,world):
        self.receipts.clear()
        return self.env.reset(world)
    def step(self,action):
        nxt=self.env.step(action)
        tr=nxt['transition'];visible=[v for v in tr['trace'] if v['tick']<=self.horizon]
        self.receipts.append(dict(start_tick=tr['before']['tick'],end_tick=visible[-1]['tick'],
            water_before=tr['before']['thirst'],water_after=visible[-1]['thirst'],correct=outcome(tr)['correct'],
            terminal=visible[-1]['dead'],transition_sha=digest(tr)))
        return nxt

def annotate(game,mix,receipts,spec):
    first_start=first_confirm=None
    assert len(receipts)==game['decisions']
    for i,(r,v) in enumerate(zip(game['rows'],receipts)):
        assert v['transition_sha']==r['transition_sha'] and (r['tick'],r['end_tick'])==(v['start_tick'],v['end_tick'])
        drank=r['action']['verb']=='drink' and v['water_after']>v['water_before']
        if drank and first_start is None:
            first_start=r['tick'];first_confirm=r['end_tick']
        eligible=(i+1)%spec['stride']==0 and not (v['terminal'] or r['cut'])
        sign=(1 if v['correct'] else -1) if eligible else 0
        scaled,contribution,total=mix.score(r['reward'],sign,r['end_tick']-r['tick'])
        assert abs(contribution)<=spec['cap']*abs(r['reward'])+1e-15
        r.update(water_before=v['water_before'],water_after=v['water_after'],drink_success=drank,feedback_label=sign,
                 r2_scaled=scaled,contribution=contribution,total_reward=total)
    game.update(first_drink_start=first_start,first_drink_confirmed=first_confirm,
                wait_to_drink_60=spec['horizon'] if first_start is None else first_start)

def main(seed,arm):
    spec=read(HERE/'design.json');train=read(TRAIN/'design.json');ctx=read(TRAIN/'context.json')
    assert spec['runner_sha']==sha(ARCHIVE/'freeze.py') and spec['training_algorithm_sha']==sha(TRAIN/'fixed_context_ppo.py')
    assert arm in spec['arms'] and seed in spec['seeds'] and digest(ctx)==spec['context_sha']
    model=TRAIN/'models'/str(seed)/arm
    reg=read(model/'registration.json');expected=read(model/'update.json')['after']
    assert sha(model/'checkpoint/manifest.json')==spec['checkpoints'][f'{seed}/{arm}']
    mix=ReplayRewardMix.from_study(train,GAMMA,arm=='ordered')
    assert (mix.alpha,mix.beta,mix.scale_fraction)==(reg['alpha'],reg['beta'],.1)
    torch.manual_seed(6100000)
    config=reg['config']|read(TRAIN/'model_config.json')|{'rng_seed':6100000}
    policy=FixedContextPolicy(SimpleNamespace(**config),arm,ctx,mix)
    assert state(policy)==reg['initial']
    load(policy,model/'checkpoint')
    assert state(policy)==expected
    folder=HERE/'runs'/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),GodotEnv(str(GODOT),str(ARCHIVE/'severe/audit-game'),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(ARCHIVE/'severe/plain-game'),map_mode='infinite_freeze',navigation_steps=4) as plain:
        observer=ObservedGame(plain,spec['horizon'])
        for world in spec['worlds']:
            out=folder/f'{world}.json'
            if out.exists():
                old=read(out)
                assert old['design_sha']==digest(spec) and old['frozen']==expected and old['game']['world']==world and old['game']['mode']==arm
                continue
            game=play(policy,env,observer,spec,world,arm)
            annotate(game,mix,observer.receipts,spec)
            tmp=out.with_suffix('.partial')
            tmp.write_text(json.dumps(dict(seed=seed,design_sha=digest(spec),frozen=expected,game=game),separators=(',',':')))
            tmp.replace(out)
    assert state(policy)==expected
    (folder/'frozen.json').write_text(json.dumps(dict(seed=seed,arm=arm,state=expected,worlds=len(spec['worlds']),design_sha=digest(spec))))
    print('complete',seed,arm,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
