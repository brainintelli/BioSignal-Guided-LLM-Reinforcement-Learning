"""Replay registered checkpoints with an archived trace and a fixed virtual-time horizon."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent
sys.path.insert(0,str(STUDY))
from resources import GODOT,PROJECT
from train import FixedContextPolicy
import torch
from agent.m9_env import GodotEnv
from agent.m9_contract import Reward,potential
from agent.m9_policy import load
from agent.mea_long_ablation import ORIGINAL_TENSORS
from mea_state2_r2.features import digest,NAMES
from mea_state2_r2.protocol import GAMMA
from tools.mea_shuffle_train import state

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def score_step(tr,obs,tracker,env,horizon):
    old=tracker.previous
    actual=tracker.score(tr,obs)['reward']
    trace=[v for v in tr['trace'] if v['tick']<=horizon]
    assert trace and [v['tick'] for v in trace]==list(range(tr['before']['tick']+1,tr['before']['tick']+len(trace)+1))
    if len(trace)==len(tr['trace']):
        boundary=obs
    else:
        boundary=env.call('landmark')['landmark']
    end=trace[-1]; dt=len(trace)
    assert end['tick']==boundary['world_tick'] and end['visited_count']==boundary['visited_count']
    # Godot exposes player reserves rounded to 0.1 in observation; trace retains full precision.
    assert all(abs(end[k]-boundary['player'][k])<=0.050001 for k in ('health','hunger','thirst')), (end,boundary['player'])
    living=sum(GAMMA**i*.01*(not v['dead']) for i,v in enumerate(trace))
    reward=living+GAMMA**dt*potential(boundary,'bottleneck',end['dead'])-old
    if len(trace)==len(tr['trace']):assert math.isclose(actual,reward,abs_tol=1e-11)
    return reward,living,end,len(trace)<len(tr['trace'])

def play(policy,env,plain,spec,world,mode):
    first=env.reset(world);shadow=plain.reset(world)
    assert digest(first['observation'])==digest(shadow['observation'])
    obs=first['observation']
    assert first['starting_scene']==spec.get('scene_name','supplied_95') and all(obs['player'][k]==v for k,v in spec['scenes'].items())
    tracker=Reward(obs,'bottleneck',GAMMA); visited=obs['visited_count'];rows=[];total=0.;alive_score=0.
    schedule=spec['shuffle'][str(world)] if mode=='trained_shuffled' else list(range(spec['horizon']))
    done=False
    while obs['world_tick']<spec['horizon'] and not done:
        t=obs['world_tick'];w=spec['windows'][schedule[t]];prep,snap=policy.replay_input(obs,w)
        vec=[0.]*4 if mode in ('control_masked','trained_masked') else w['normalized']
        fmt=','.join(f'{n}={v:.3f}' for n,v in zip(NAMES,vec))
        assert prep['wetware_features']==vec==prep['critic_input'][-4:] and fmt in prep['messages'][1]['content']
        assert snap['source_pin']==spec['source_sha'] and snap['window_index']==w['index']
        torch.manual_seed(world*100+t)
        act=policy.act([prep],with_reference=False)['actions'][0]
        n=env.step(act); ck=plain.step(act)
        assert digest(n['transition'])==digest(ck['transition']) and digest(n['observation'])==digest(ck['observation'])
        rew,living,end,cut=score_step(n['transition'],n['observation'],tracker,env,spec['horizon'])
        total+=rew;alive_score+=living
        rows.append(dict(tick=t,window=w['index'],events=w['events_sha'],features=vec,action=act,transition_sha=digest(ck['transition']),end_tick=end['tick'],reward=rew,cut=cut))
        obs=n['observation']; done=n['terminated']
    assert end['tick']==spec['horizon'] or end['dead']
    return dict(world=world,mode=mode,decisions=len(rows),end_tick=end['tick'],survived=int(not end['dead']),
        r1=total,r1_per_tick=total/end['tick'],living=alive_score,shape=total-alive_score,new_cells=end['visited_count']-visited,
        health=end['health'],thirst=end['thirst'],hunger=end['hunger'],cut=cut,rows=rows)

def main(seed,folder=ROOT):
    spec=read(folder/'design.json');assert seed in spec['seeds'];out=folder/'runs'/str(seed);out.mkdir(parents=True,exist_ok=True)
    ref=STUDY/'pre-stim-baseline/models'/str(seed)/'baseline';mask=STUDY/'models'/str(seed)/'masked'
    reg=read(ref/'registration.json')
    torch.manual_seed(6100000)
    ctx=dict(read(STUDY/'pre-stim-baseline/baseline.json'),source_sha256=spec['source_sha'],state2_source='archived_silent_recording_electrode_remapped')
    cfg=reg['config']|read(STUDY/'model_config.json')|dict(rng_seed=6100000)
    assert ctx['normalization']==spec['normalization'] and ctx['roi_sha']==spec['roi_sha']
    policy=FixedContextPolicy(SimpleNamespace(**cfg),'baseline',ctx)
    assert state(policy)==ORIGINAL_TENSORS==reg['initial']
    modes=(['initial_ordered'] if seed==spec['seeds'][0] else [])+['trained_ordered','trained_shuffled','trained_masked','control_masked']
    if folder!=ROOT:modes=spec['modes']
    original_project=PROJECT if folder==ROOT else folder/'plain-game'
    with torch.no_grad(),GodotEnv(str(GODOT),str(folder/'audit-game'),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(original_project),map_mode='infinite_freeze',navigation_steps=4) as plain:
        for mode in modes:
            policy.arm='A' if mode in ('trained_masked','control_masked') else 'C'
            ex=ref if mode!='control_masked' else mask
            if mode in ('trained_ordered','control_masked'):
                policy.comparison='masked' if mode=='control_masked' else 'baseline'
                load(policy,ex/'checkpoint')
            expected=reg['initial'] if mode=='initial_ordered' else read(ex/'update.json')['after']
            assert state(policy)==expected
            # Completed files are immutable; interrupted modes restart from the first world.
            dest=out/(mode+'.json')
            if dest.exists():
                prev=read(dest);assert prev['study_sha']==digest(spec) and [r['world'] for r in prev['games']]==spec['worlds']
                continue
            games=[]
            for world in spec['worlds']:
                games.append(play(policy,env,plain,spec,world,mode))
                print(seed,mode,world,len(games),flush=True)
            assert state(policy)==expected
            tmp=out/(mode+'.partial');tmp.write_text(json.dumps(dict(seed=seed,mode=mode,study_sha=digest(spec),frozen=expected,games=games),separators=(',',':')))
            tmp.replace(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('--stress',action='store_true');p.add_argument('--severe',action='store_true');a=p.parse_args();main(a.seed,ROOT/'severe' if a.severe else ROOT/'stress' if a.stress else ROOT)
