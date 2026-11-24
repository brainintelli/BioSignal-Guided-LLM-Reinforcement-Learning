"""Verify the prespecified risk in an unused development world with the native simulator."""
import json
from pathlib import Path
from agent.m9_env import GodotEnv
from resources import GODOT
folder=Path(__file__).resolve().parent
world=179999999
assert world not in json.loads((folder/'design.json').read_text())['worlds']
with GodotEnv(str(GODOT),str(folder/'plain-game'),map_mode='infinite_freeze',navigation_steps=4) as env:
    obs=env.reset(world)['observation']
    assert obs['player']['thirst']==1.6 and obs['player']['health']==25
    for _ in range(60):
        action=next(a for a in obs['allowed_actions'] if a['verb']=='wait')
        new=env.step(action);obs=new['observation']
        if new['terminated']:break
    result=dict(development_world=world,action='wait',died=new['terminated'],end_tick=obs['world_tick'],end_health=obs['player']['health'])
    assert result['died'] and 20<result['end_tick']<60
    (folder/'idle-check.json').write_text(json.dumps(result));print(result)
