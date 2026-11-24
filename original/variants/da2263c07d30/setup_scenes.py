"""Create isolated start configurations and verify them on development maps only."""
from pathlib import Path
import json,hashlib,shutil,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from resources import PROJECT,GODOT
from agent.m9_env import GodotEnv
from mea_state2_r2.features import digest
ARCH=Path.home()/'mea-22021-20261118-mea-stim-ablation/archive-frozen-60tick'
def read(p):return json.loads(p.read_bytes())
def manifest(path):
    files=[path/'project.godot']+[f for d in ('scripts','configs','godot_tests') for f in (path/d).rglob('*') if f.is_file()]
    return {f.relative_to(path).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
def build(name,res):
    pins={}
    audit_source=ARCH/'audit-game' if (ARCH/'audit-game').exists() else PROJECT
    for role,src in [('plain',PROJECT),('audit',audit_source)]:
        dst=HERE/'scenes'/name/role
        dst.mkdir(parents=True,exist_ok=False)
        old=manifest(src)
        for d in ('scripts','configs','godot_tests'):shutil.copytree(src/d,dst/d)
        shutil.copyfile(src/'project.godot',dst/'project.godot')
        script=dst/'scripts/mea_supplied_session.gd'
        st=script.read_text()
        before='\t\tworld.hunger = 95.0\n\t\tworld.thirst = 95.0'
        assert st.count(before)==1 and st.count('"supplied_95"')==1
        after='\n'.join(f'\t\tworld.{k} = {float(v)}' for k,v in res.items())
        script.write_text(st.replace(before,after).replace('"supplied_95"',f'"stress_{name}"'))
        new=manifest(dst);assert new.keys()==old.keys() and [k for k in new if new[k]!=old[k]]==['scripts/mea_supplied_session.gd']
        pins[role]=digest(new)
    return pins
def pilot(name,res,world,plan,env,plain):
    first=env.reset(world);check=plain.reset(world)
    assert first['observation']==check['observation'] and first['starting_scene']=='stress_'+name
    obs=first['observation'];assert all(obs['player'][k]==v for k,v in res.items())
    acted=False;events=[];last=None
    while obs['world_tick']<60:
        legal=obs['allowed_actions'];a={'verb':'wait'}
        if not acted and plan=='eat' and {'verb':'eat'} in legal:a={'verb':'eat'}
        if not acted and plan=='drink':
            if {'verb':'drink'} in legal:a={'verb':'drink'}
            else:
                candidates=[(v['path_steps'],v['id']) for v in obs['nearby'] if v['kind']=='water' and v['available'] and {'verb':'navigate','target_id':v['id']} in legal]
                if candidates:a={'verb':'navigate','target_id':min(candidates)[1]}
        n=env.step(a);check=plain.step(a);tr=n['transition'];assert tr==check['transition'] and n['observation']==check['observation']
        visible=[v for v in tr['trace'] if v['tick']<=60];last=visible[-1];obs=n['observation']
        if a['verb'] in ('drink','eat'):
            k='thirst' if a['verb']=='drink' else 'hunger'
            assert last[k]>tr['before'][k];acted=True;events.append([a['verb'],last['tick'],tr['before'][k],last[k]])
        if last['dead']:break
    return dict(world=world,policy=plan,end_tick=last['tick'],survived=not last['dead'],events=events)
def main():
    scenes=read(HERE/'scenes.json');pins={};tests={}
    worlds=[246101007,246101123,246101311,246101411]
    for name,res in scenes.items():
        pins[name]=build(name,res);tests[name]=[]
        rd=HERE/'scenes'/name
        with GodotEnv(str(GODOT),str(rd/'audit'),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(rd/'plain'),map_mode='infinite_freeze',navigation_steps=4) as plain:
            for world in worlds:
                for behavior in ('wait','drink','eat'):
                    tests[name].append(pilot(name,res,world,behavior,env,plain))
    (HERE/'scene-check.json').write_text(json.dumps(dict(pins=pins,worlds=worlds,tests=tests),indent=2))
    for k,l in tests.items():print(k,[(v['policy'],v['end_tick'],v['survived']) for v in l[:3]])
if __name__=='__main__':main()
