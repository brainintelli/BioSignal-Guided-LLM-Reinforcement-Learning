"""Recover all six original scene pins and repeat the original development-map checks."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;STRESS=ROOT/'stress'
BASE=Path.home()/'mea-state2-r2-20261118/run02/project'
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'standalone/mea_state2_r2'));sys.path.insert(0,str(ROOT));sys.path.insert(0,str(STRESS))
from setup_scenes import pilot,manifest,GodotEnv,GODOT,digest


def main():
    original=HERE/'original-scene-check.json'; raw=original.read_bytes();prior=json.loads(raw);actual=json.loads((STRESS/'scene-check.json').read_bytes());design=json.loads((STRESS/'design.json').read_bytes())
    before=json.loads((STRESS/'design-repair.json').read_bytes())['pilot_sha_before'];assert hashlib.sha256(raw).hexdigest()==before
    assert prior['worlds']==actual['worlds'] and prior['tests']==actual['tests']
    uids={f.relative_to(BASE):f for sub in ['scripts','configs','godot_tests'] for f in (BASE/sub).rglob('*.uid')};assert len(uids)==39
    recreated={};tested={}
    for scene,params in design['scenes'].items():
        recreated[scene]={}
        paths={}
        for role,source in [('plain',STRESS/'scenes'/scene/'plain'),('audit',HERE/'native'/scene/'audit-game')]:
            dest=HERE/'pinned-scenes'/scene/role;paths[role]=dest
            if not dest.exists():
                dest.mkdir(parents=True)
                for sub in ['scripts','configs','godot_tests']:shutil.copytree(source/sub,dest/sub)
                shutil.copy2(source/'project.godot',dest/'project.godot')
                for rel,uid in uids.items():
                    assert not (dest/rel).exists();shutil.copy2(uid,dest/rel)
            verified=manifest(dest);before_uid=manifest(source)
            assert all(verified[k]==v for k,v in before_uid.items())
            assert set(verified)-set(before_uid)=={p.as_posix() for p in uids}
            recreated[scene][role]=digest(verified);assert recreated[scene][role]==prior['pins'][scene][role]
        tested[scene]=[]
        with GodotEnv(str(GODOT),str(paths['audit']),map_mode='infinite_freeze',navigation_steps=4) as env,GodotEnv(str(GODOT),str(paths['plain']),map_mode='infinite_freeze',navigation_steps=4) as plain:
            for world in prior['worlds']:
                for plan in ['wait','drink','eat']:
                    tested[scene].append(pilot(scene,params,world,plan,env,plain))
        assert tested[scene]==prior['tests'][scene]
    replayed=json.dumps(dict(pins=recreated,worlds=prior['worlds'],tests=tested),indent=2).encode()
    assert replayed==raw
    report=dict(original_scene_check_sha256=before,reproduced_file_sha256=hashlib.sha256(replayed).hexdigest(),restored_scene_pins=recreated,uid_files_per_game=len(uids),shared_executable_script_bytes_with_native_replay=True,pilot_scenarios=sum(map(len,tested.values())),development_maps=prior['worlds'],different_fields_between_evaluated_and_original_pilot=['pins'],verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'source-recovery.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
