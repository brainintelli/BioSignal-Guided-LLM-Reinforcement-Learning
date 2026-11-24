"""Independent native-game replay and input, reward, checkpoint and budget checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import math
from resources import GODOT, PROJECT
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from mea_state2_r2.features import NAMES,digest
from mea_state2_r2.protocol import GAMMA
from mea_state2_r2.trajectory import audit

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def check(s,ctx,seed,arm):
    d=HERE/'models'/str(seed)/arm
    reg=read(d/'registration.json');up=read(d/'update.json');man=read(d/'checkpoint/manifest.json');meta=read(d/'checkpoint/policy.json')
    assert reg['study_digest']==digest(s) and reg['windows_digest']==digest(ctx['series']) and reg['source_sha']==ctx['source_sha256']
    assert reg['arm']==arm and reg['training_seed']==seed and reg['initial']==up['initial']
    assert man['actor_sha256']==up['after']['actor'] and man['registration_sha256']==digest(reg)
    assert all(sha(d/'checkpoint'/name)==h for name,h in man['files'].items())
    assert not meta['r2_enabled'] and not meta['realtime_training'] and not meta['stimulation'] and meta['beta']==0
    assert up['actor_steps']==[8] and up['critic_steps']==[64] and man['update']==4
    perm=s['training_order'][str(seed)] if arm=='shuffled' else list(range(48))
    count=0;positives=0;seen=[]
    with GodotEnv(str(GODOT),str(PROJECT),map_mode='infinite_freeze',navigation_steps=4) as env:
        for j,world in enumerate(s['episodes'][str(seed)]):
            init=env.reset(world);obs=init['observation']
            assert init['starting_scene']==s['starting_scene'] and all(obs['player'][k]==v for k,v in s['starting_reserves'].items())
            tracker=Reward(obs,'bottleneck',GAMMA)
            rows=[json.loads(v) for v in (d/f'train-{j+1}.jsonl').read_text().splitlines()]
            rr=up['rounds'][j];assert len(rows)==12 and rr['world']==world and rr['update']['optimizer_steps']==2
            audit(rows,rr['bootstrap'],GAMMA,reg['config']['gae_lambda'])
            for i,r in enumerate(rows):
                w=ctx['series']['train'][perm[j*12+i]];sn=r['snapshot'];pr=r['prepared']
                vec=w['normalized'] if arm!='masked' else [0.]*4
                fmt=','.join(f'{n}={v:.3f}' for n,v in zip(NAMES,vec))
                assert pr['wetware_features']==pr['critic_input'][-4:]==vec and fmt in pr['messages'][1]['content']
                assert sn['window_index']==w['index'] and sn['events_sha256']==w['events_sha'] and sn['source_pin']==ctx['source_sha256']
                assert r['observation']==obs and r['seed']==world and r['index']==i
                nxt=env.step(r['action']);tr=nxt['transition'];obs=nxt['observation']
                assert tr==r['transition'] and obs==r['next_observation']
                native=tracker.score(tr,obs)['reward']
                assert math.isclose(native,r['reward'],abs_tol=1e-12) and r['contribution']==0 and r['reward']==r['r1_score']['reward']
                expected=int(r['action']['verb']=='drink' and tr['before']['thirst']<=20 and tr['after']['thirst']>tr['before']['thirst'])
                assert r['r2']==expected
                seen.append(w['index']);positives+=expected;count+=1
    assert count==48 and len(set(seen))==48
    return dict(seed=seed,arm=arm,steps=count,positive_log_only=positives,reward_contribution=0,initial=up['initial'],
        after=up['after'],config_digest=digest(reg['config']),actor_steps=8,critic_steps=64,checkpoint_manifest=sha(d/'checkpoint/manifest.json'))

def main():
    s=read(HERE/'design.json');ctx=read(HERE/'context.json')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(check,s,ctx,i,a) for i in s['training_seeds'] for a in s['arms']]
        audits=[j.result() for j in jobs]
    assert len({digest(r['initial']) for r in audits})==len({r['config_digest'] for r in audits})==1
    (HERE/'training-audit.json').write_text(json.dumps(audits,indent=2))
    print('verified',len(audits),'models;',sum(r['steps'] for r in audits),'transitions')
if __name__=='__main__':main()
