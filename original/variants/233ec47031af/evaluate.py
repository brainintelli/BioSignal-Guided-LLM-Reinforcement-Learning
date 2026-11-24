"""Paired native-game evaluation of already-fitted policies under resource stress."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
import torch
from agent.m9_env import GodotEnv
from agent.m9_policy import load
from mea_state2_r2.features import digest
from play60 import play
from resources import GODOT
from tools.mea_shuffle_train import state
from train import FixedContextPolicy,read
from setup_scenes import manifest

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class ResourceTrace:
    def __init__(self,env):
        self.env=env
    def reset(self,world):
        init=self.env.reset(world)
        self.actions=[];self.obs=init['observation']
        return init
    def step(self,action):
        assert action in self.obs['allowed_actions']
        nxt=self.env.step(action);tr=nxt['transition']
        visible=[v for v in tr['trace'] if v['tick']<=60]
        record={'start':tr['before']['tick'],'end':visible[-1]['tick'],'water0':tr['before']['thirst'],'food0':tr['before']['hunger'],
                'water1':visible[-1]['thirst'],'food1':visible[-1]['hunger'],'drink':None,'eat':None,'tr_sha':digest(tr)}
        for verb,key in [('drink','thirst'),('eat','hunger')]:
            if action['verb']==verb:
                record[verb]=next((v['tick'] for v in visible if v[key]>tr['before'][key]),None)
        self.actions.append(record);self.obs=nxt['observation'];return nxt

def append_resources(game,observer):
    assert len(observer.actions)==len(game['rows'])
    first={'drink':None,'eat':None}
    for event,row in zip(observer.actions,game['rows']):
        assert row['transition_sha']==event['tr_sha'] and (row['tick'],row['end_tick'])==(event['start'],event['end'])
        row['resource_event']=event
        for verb in first:
            if event[verb] is not None and first[verb] is None:first[verb]=event[verb]
    game['first_success_tick']=first
    game['restricted_wait']={k:(60 if v is None else v) for k,v in first.items()}

def main(seed,arm,scene):
    s=read(HERE/'design.json');old=read(PARENT/'design.json')
    assert arm in s['arms'] and seed in s['training_seeds'] and scene in s['scenes']
    assert sha(HERE/'evaluate.py')==s['runner_sha'] and digest(old)==s['training_design_sha'] and sha(PARENT/'play60.py')==old['eval_code_sha']==s['play_sha']
    assert sha(PARENT/'train.py')==old['training_code_sha'] and sha(PARENT/'training-audit.json')==s['training_audit_sha']
    model=PARENT/'models'/str(seed)/arm;reg=read(model/'registration.json');up=read(model/'update.json');man=read(model/'checkpoint/manifest.json')
    assert reg['arm']==arm and reg['training_seed']==seed and reg['study_digest']==digest(old)
    frozen=s['states'][f'{seed}/{arm}'];assert up['after']==frozen and man['actor_sha256']==frozen['actor']
    assert all(sha(model/'checkpoint'/n)==v for n,v in man['files'].items())
    ctx=read(PARENT/'context.json');assert digest(ctx)==old['context_sha'] and ctx['normalization']==s['normalization'] and ctx['roi_sha']==s['roi_sha']
    torch.manual_seed(6100000);cfg=reg['config']|read(PARENT/'model_config.json')|{'rng_seed':6100000}
    policy=FixedContextPolicy(SimpleNamespace(**cfg),arm,ctx)
    assert state(policy)==reg['initial'];load(policy,model/'checkpoint');assert state(policy)==frozen
    policy.context=ctx|dict(source_sha256=s['source_sha'],state2_source='archived_prestimulation_quiet_recording')
    game_spec=s|dict(starting_scene='stress_'+scene,starting_reserves=s['scenes'][scene])
    path=HERE/'scenes'/scene
    assert all(digest(manifest(path/k))==v for k,v in s['game_pins'][scene].items())
    out=HERE/'games'/str(seed)/scene/arm;out.mkdir(parents=True,exist_ok=True);pin=digest(s)
    with torch.inference_mode(),GodotEnv(str(GODOT),str(path/'audit'),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(path/'plain'),map_mode='infinite_freeze',navigation_steps=4) as plain:
        observer=ResourceTrace(plain)
        for world in s['worlds']:
            f=out/f'{world}.json'
            if f.exists():
                d=read(f);assert d['spec']==pin and d['frozen']==frozen and d['game']['world']==world and d['scene']==scene and d['game']['mode']==arm
                continue
            game=play(policy,env,observer,game_spec,world,arm)
            append_resources(game,observer)
            tmp=f.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(scene=scene,seed=seed,frozen=frozen,spec=pin,game=game),separators=(',',':')))
            tmp.replace(f)
    assert state(policy)==frozen and len(list(out.glob('[0-9]*.json')))==len(s['worlds'])
    (out/'frozen.json').write_text(json.dumps(dict(scene=scene,seed=seed,arm=arm,state=frozen,worlds=len(s['worlds']),spec=pin)))
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2],sys.argv[3])
