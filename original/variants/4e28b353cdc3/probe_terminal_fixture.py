"""Native legal-wait prefix for terminal branch coverage; no state injection."""
import hashlib,json,os
from pathlib import Path
from agent.m9_env import GodotEnv
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
work=Path(os.environ['TASK_ROOT']);runtime=Path(os.environ['TASK_RUNTIME'])
seed=4126101;trace=[]
with GodotEnv(str(runtime/'godot/Godot_v4.7.2-stable_linux.x86_64'),str(work/'project'),map_mode='infinite_freeze',navigation_steps=4) as env:
    obs=env.reset(seed)['observation']
    for i in range(2000):
        before=obs;nxt=env.step({'verb':'wait'});obs=nxt['observation']
        if nxt['terminated']:
            result=dict(seed=seed,source='native_legal_wait_prefix',scripted_actions_in_ppo=False,prefix_steps=i,
                prefix_transition_hashes=trace,last_alive_observation=before,
                natural_death_transition=nxt['transition'],natural_death_observation=obs)
            break
        trace.append(digest(nxt['transition']))
    else:raise RuntimeError('No natural terminal within fixed2000wait budget')
with (work/'terminal_fixture.json').open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(prefix_steps=result['prefix_steps'],last_alive_tick=before['tick'],player=before['player'],death_tick=obs['tick']),ensure_ascii=False))
