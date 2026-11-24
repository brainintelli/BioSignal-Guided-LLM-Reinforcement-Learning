"""Confirm both signs at the first eligible feedback slot using disjoint design seeds."""
import json
import sys
from pathlib import Path

from agent.m9_env import GodotEnv
from livepair.predictability import outcome

root, godot, project = map(Path,sys.argv[1:])
study=json.loads((root/'design.json').read_text())
records=[]
with GodotEnv(str(godot),str(project),map_mode='infinite_freeze',navigation_steps=4) as env:
    for seed in study['pilot_seeds']:
        def eligible_state():
            start=env.reset(seed)
            if start.get('starting_scene')!=study['starting_scene']:
                raise ValueError('Unexpected scene')
            if any(start['observation']['player'][k]!=v for k,v in study['starting_reserves'].items()):
                raise ValueError('Incorrect reserves')
            for _ in range(2):
                reply=env.step({'verb':'wait'})
            return reply['observation']
        obs=eligible_state()
        alternatives=[{'verb':'wait'}]+[a for a in obs['allowed_actions'] if a['verb']=='move']
        for action in alternatives:
            if eligible_state()!=obs:
                raise ValueError('Paired counterfactual state changed')
            obs_result=outcome(env.step(action)['transition'])
            records.append(dict(seed=seed,step=3,action=action,**obs_result))
        if {r['correct'] for r in records if r['seed']==seed}!={True,False}:
            raise ValueError(f'Both classes are not accessible from pilot world {seed}')
(root/'pilot.json').write_text(json.dumps(records,indent=2))
print('Pilot worlds with both signs:',len(study['pilot_seeds']),flush=True)
