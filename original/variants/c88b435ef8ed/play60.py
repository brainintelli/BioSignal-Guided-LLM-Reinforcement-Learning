from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import sys
import torch
from agent.m9_env import GodotEnv
from agent.m9_contract import Reward,potential
from agent.m9_policy import load
from agent.mea_long_ablation import ORIGINAL_TENSORS
from mea_state2_r2.features import NAMES,digest
from mea_state2_r2.protocol import GAMMA
from tools.mea_shuffle_train import state
from train import FixedContextPolicy,read
from resources import PROJECT,GODOT
HERE=Path(__file__).resolve().parent
ARCHIVE=Path.home()/'mea-22021-20261118-mea-stim-ablation/archive-frozen-60tick'

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
    assert first['starting_scene']==spec['starting_scene'] and all(obs['player'][k]==v for k,v in spec['starting_reserves'].items())
    tracker=Reward(obs,'bottleneck',GAMMA); visited=obs['visited_count'];rows=[];total=0.;alive_score=0.
    schedule=spec['shuffle'][str(world)] if mode=='shuffled' else list(range(spec['horizon']))
    done=False
    while obs['world_tick']<spec['horizon'] and not done:
        t=obs['world_tick'];w=spec['windows'][schedule[t]];prep,snap=policy.replay_input(obs,w)
        vec=[0.]*4 if mode=='masked' else w['normalized']
        fmt=','.join(f'{n}={v:.3f}' for n,v in zip(NAMES,vec))
        assert prep['wetware_features']==vec==prep['critic_input'][-4:] and fmt in prep['messages'][1]['content']
        assert snap['source_pin']==spec['source_sha'] and snap['window_index']==w['index']
        torch.manual_seed(world*100+t)
        act=policy.act([prep],with_reference=False)['actions'][0]
        n=env.step(act); ck=plain.step(act)
        assert digest(n['transition'])==digest(ck['transition']) and digest(n['observation'])==digest(ck['observation'])
        rew,living,end,cut=score_step(n['transition'],n['observation'],tracker,env,spec['horizon'])
        total+=rew;alive_score+=living
        rows.append(dict(tick=t,window=w['index'],events=w['events_sha'],features=vec,action=act,transition_sha=digest(ck['transition']),end_tick=end['tick'],reward=rew,cut=cut,new_cells_at_end=end['visited_count']-visited,
                         r2_logged=int(not cut and act['verb']=='drink' and ck['transition']['before']['thirst']<=20 and end['thirst']>ck['transition']['before']['thirst']),r2_contribution=0))
        obs=n['observation']; done=n['terminated']
    assert end['tick']==spec['horizon'] or end['dead']
    return dict(world=world,mode=mode,decisions=len(rows),end_tick=end['tick'],survived=int(not end['dead']),
        r1=total,r1_per_tick=total/end['tick'],living=alive_score,shape=total-alive_score,new_cells=end['visited_count']-visited,
        health=end['health'],thirst=end['thirst'],hunger=end['hunger'],cut=cut,rows=rows)

def main(seed,arm):
    spec=read(HERE/'design.json');ctx=read(HERE/'context.json')
    model=HERE/'models'/str(seed)/arm
    reg=read(model/'registration.json');up=read(model/'update.json')
    assert reg['study_digest']==digest(spec) and reg['windows_digest']==digest(ctx['series'])
    assert up['actor_steps']==[8] and up['critic_steps']==[64] and up['initial']==reg['initial']==ORIGINAL_TENSORS
    assert len(up['rounds'])==4 and all(v['contribution']==0 and v['decisions']==12 for v in up['rounds'])
    assert digest(ctx)==spec['context_sha'] and spec['alpha']==1 and spec['beta']==0
    for f,k in [('train.py','training_code_sha'),('play60.py','eval_code_sha')]:
        assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==spec[k]
    torch.manual_seed(6100000)
    cfg=reg['config']|read(HERE/'model_config.json')|dict(rng_seed=6100000)
    policy=FixedContextPolicy(SimpleNamespace(**cfg),arm,ctx)
    assert state(policy)==reg['initial']
    load(policy,model/'checkpoint');frozen=state(policy)
    assert frozen==up['after']
    dest=HERE/'games'/str(seed)/arm;dest.mkdir(parents=True,exist_ok=True)
    pin=digest(spec)
    with torch.no_grad(),GodotEnv(str(GODOT),str(ARCHIVE/'audit-game'),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(PROJECT),map_mode='infinite_freeze',navigation_steps=4) as plain:
        for world in spec['worlds']:
            f=dest/f'{world}.json'
            if f.exists():
                v=read(f);assert v['frozen']==frozen and v['spec']==pin and v['game']['world']==world and v['game']['mode']==arm
                continue
            game=play(policy,env,plain,spec,world,arm)
            tmp=f.with_suffix('.partial')
            tmp.write_text(json.dumps(dict(seed=seed,frozen=frozen,spec=pin,game=game),separators=(',',':')))
            tmp.replace(f)
    assert state(policy)==frozen and len(list(dest.glob('[0-9]*.json')))==48
    (dest/'frozen.json').write_text(json.dumps(dict(seed=seed,arm=arm,worlds=48,state=frozen,spec=pin)))
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
