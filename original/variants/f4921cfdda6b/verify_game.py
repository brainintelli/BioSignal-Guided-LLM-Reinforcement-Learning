"""Independently replay saved holdout actions in the Godot simulation."""
import json
import math
import sys
from pathlib import Path

import torch
from agent.m9_contract import Reward
from agent.m9_env import GodotEnv
from agent.mea_contract import observation_key
from livepair.predictability import outcome
from mea_state2_r2.protocol import GAMMA

root=Path(__file__).resolve().parent
study=json.loads((root/'design.json').read_text())
summary=[]
with GodotEnv(sys.argv[1],str(root/'game'),map_mode='infinite_freeze',navigation_steps=4) as env:
    for s in study['training_seeds']:
        for arm in study['arms']:
            directory=root/'models'/str(s)/arm
            result=json.loads((directory/'result.json').read_text())
            states=torch.load(directory/'checkpoint/optimizers.pt',map_location='cpu',weights_only=True)
            steps={name:sorted({int(x['step']) for x in states[name]['state'].values()}) for name in ('actor','critic')}
            if steps!={'actor':[2],'critic':[16]}:
                raise ValueError(f'Incorrect optimizer step counts: {steps}')
            totals={'positive':0,'negative':0,'silent':0}
            count=0
            for metric in result['eval']:
                seed=metric['seed']
                rows=json.loads((directory/f'eval-{seed}.json').read_text())
                start=env.reset(seed)
                obs=start['observation']
                if start['starting_scene']!=study['starting_scene'] or any(obs['player'][k]!=v for k,v in study['starting_reserves'].items()):
                    raise ValueError('Wrong starting fields')
                visited=obs['visited_count']
                tracker=Reward(obs,'bottleneck',GAMMA)
                score=0.
                for i,row in enumerate(rows):
                    if row['observation_key']!=observation_key(obs):
                        raise ValueError('Holdout context differs')
                    nxt=env.step(row['action'])
                    obs=nxt['observation']
                    tr=nxt['transition']
                    r1=tracker.score(tr,obs)['reward']
                    sign=(1 if outcome(tr)['correct'] else -1) if (i+1)%3==0 and not nxt['terminated'] else 0
                    if tr['after']!=row['after'] or sign!=row['simulated_r2'] or not math.isclose(r1,row['r1'],abs_tol=1e-12):
                        raise ValueError('Holdout trace or scoring differs')
                    totals[{1:'positive',-1:'negative',0:'silent'}[sign]]+=1
                    score+=r1
                    count+=1
                if metric['new_cells']!=obs['visited_count']-visited or not math.isclose(score,metric['r1_sum'],abs_tol=1e-12):
                    raise ValueError('Holdout endpoints differ')
            summary.append(dict(training_seed=s,arm=arm,optimizer_steps=steps,worlds=len(result['eval']),actions=count,feedback=totals))
            print(s,arm,count,flush=True)
(root/'simulation-audit.json').write_text(json.dumps(summary,indent=2))
