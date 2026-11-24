"""Replay every training action; bind the finished paired checkpoints to the locked test plan."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent
sys.path.insert(0,str(TRAIN))
from resources import GODOT
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from mea_state2_r2.features import digest
from mea_state2_r2.trajectory import audit
GAME=Path.home()/'mea-22021-20261118-mea-stim-ablation/archive-frozen-60tick/severe/plain-game'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(seed,arm,s,ctx):
    folder=TRAIN/'models'/str(seed)/arm;reg=read(folder/'registration.json');up=read(folder/'update.json')
    manifest=read(folder/'checkpoint/manifest.json');meta=read(folder/'checkpoint/policy.json')
    assert reg['study_digest']==digest(s) and reg['source_sha']==ctx['source_sha256'] and reg['windows_digest']==digest(ctx['series'])
    assert reg['arm']==arm and reg['training_seed']==seed and reg['feedback_rule']==s['feedback_rule']==meta['feedback_rule']
    assert up['initial']==reg['initial'] and up['after']['actor']==manifest['actor_sha256']
    assert manifest['registration_sha256']==digest(reg) and manifest['update']==s['updates']
    assert up['actor_steps']==[8] and up['critic_steps']==[64] and len(up['rounds'])==4
    for name,h in manifest['files'].items():assert sha(folder/'checkpoint'/name)==h
    beta=s['beta'] if arm=='ordered' else 0.
    assert beta==reg['beta']==meta['beta'] and reg['alpha']==meta['alpha']==1
    cfg=s['reward_mix'];n=0;pos=0;nonzero=0;max_ratio=0.;sum_r1=sum_aux=0.;reset_count=0
    first=[];active_ratios=[];first_worlds=[]
    with GodotEnv(str(GODOT),str(GAME),map_mode='infinite_freeze',navigation_steps=4) as env:
        for it,world in enumerate(s['episodes'][str(seed)]):
            obs=env.reset(world);assert obs['starting_scene']==s['starting_scene']
            obs=obs['observation'];assert all(obs['player'][k]==v for k,v in s['starting_reserves'].items())
            tracker=Reward(obs,'bottleneck',cfg['gamma_tick']);active_world=world
            rows=[json.loads(l) for l in (folder/f'train-{it+1}.jsonl').read_text(encoding='utf8').splitlines()]
            rr=up['rounds'][it];assert len(rows)==12 and rr['world']==world and rr['update']['optimizer_steps']==2
            audit(rows,rr['bootstrap'],cfg['gamma_tick'],reg['config']['gae_lambda'])
            for i,r in enumerate(rows):
                assert r['seed']==active_world and r['index']==i and digest(r['observation'])==digest(obs)
                nxt=env.step(r['action']);tr=nxt['transition'];obs=nxt['observation']
                assert tr==r['transition'] and obs==r['next_observation'] and nxt['terminated']==r['terminal']
                native=tracker.score(tr,obs)['reward'];assert math.isclose(native,r['r1_score']['reward'],abs_tol=1e-12)
                before=tr['before']['thirst'];after=tr['after']['thirst']
                expected=int(r['action']['verb']=='drink' and before<=20 and after>before)
                assert expected==r['r2'] and cfg['scale_fraction']==.1 and s['feedback_rule']['water_max']==20
                dt=tr['elapsed_ticks'];assert dt==r['duration']
                scale=expected*.1*min(abs(native),cfg['reference_per_tick']*sum(cfg['gamma_tick']**k for k in range(dt)))
                part=beta*scale
                assert math.isclose(scale,r['r2_scaled'],abs_tol=1e-15) and math.isclose(part,r['contribution'],abs_tol=1e-15)
                assert math.isclose(native+part,r['reward'],abs_tol=1e-12) and abs(part)<=.05*abs(native)+1e-15
                w=ctx['series']['train'][it*12+i]
                assert w['normalized']==r['prepared']['wetware_features'] and w['events_sha']==r['snapshot']['events_sha256'] and w['index']==r['snapshot']['window_index']
                if it==0:first.append((active_world,r['action'],digest(tr),w['events_sha']))
                if expected:first_worlds.append(active_world)
                n+=1;pos+=expected;nonzero+=part>0;sum_r1+=abs(native);sum_aux+=part
                if native:
                    max_ratio=max(max_ratio,abs(part/native))
                    if expected:active_ratios.append(abs(part/native))
                if nxt['terminated'] and i+1<len(rows):
                    reset_count+=1;active_world=s['restart_worlds'][str(world)][i]
                    obs=env.reset(active_world)['observation'];tracker=Reward(obs,'bottleneck',cfg['gamma_tick'])
    assert n==48
    return dict(seed=seed,arm=arm,steps=n,positive_labels=pos,nonzero_bonuses=nonzero,positive_worlds=len(set(first_worlds)),
        abs_r1=sum_r1,sum_bonus=sum_aux,max_ratio=max_ratio,active_ratios=active_ratios,resets=reset_count,
        manifest=sha(folder/'checkpoint/manifest.json'),initial=reg['initial'],initial_rollout=digest(first),config=reg['config'],
        actor_steps=up['actor_steps'],critic_steps=up['critic_steps'])
def main():
    plan=read(HERE/'plan.json');s=read(TRAIN/'design.json');ctx=read(TRAIN/'context.json')
    assert digest(s)==plan['training_design_sha'] and digest(ctx)==plan['context_sha']
    assert sha(TRAIN/'replay_reward.py')==plan['reward_function_sha'] and sha(TRAIN/'fixed_context_ppo.py')==plan['training_algorithm_sha']
    jobs=[(i,a) for i in s['training_seeds'] for a in s['arms']]
    with ThreadPoolExecutor(max_workers=3) as pool:checks=list(pool.map(lambda j:verify(*j,s,ctx),jobs))
    by={(r['seed'],r['arm']):r for r in checks}
    for i in s['training_seeds']:
        a,b=by[i,'r1'],by[i,'ordered']
        assert all(a[k]==b[k] for k in ('config','initial','initial_rollout','actor_steps','critic_steps','steps'))
    assert len({digest(r['initial']) for r in checks})==1
    spec=plan|dict(checkpoints={f"{r['seed']}/{r['arm']}":r['manifest'] for r in checks},
         training_max_ratio={f"{r['seed']}/{r['arm']}":r['max_ratio'] for r in checks},
         training_steps={f"{r['seed']}/{r['arm']}":r['steps'] for r in checks})
    (HERE/'training-audit.json').write_text(json.dumps(checks,indent=2))
    (HERE/'design.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    print('checked all',len(checks),'checkpoints and',sum(r['steps'] for r in checks),'training actions',flush=True)
if __name__=='__main__':main()
