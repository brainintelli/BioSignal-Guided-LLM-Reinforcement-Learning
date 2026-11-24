from pathlib import Path
import shutil
from resources import PROJECT
folder=Path(__file__).resolve().parent
for name,source in [('audit-game',folder.parent/'audit-game'),('plain-game',PROJECT)]:
    dst=folder/name
    if dst.exists():raise FileExistsError(dst)
    dst.mkdir()
    for d in ('scripts','configs','godot_tests'):shutil.copytree(source/d,dst/d)
    shutil.copyfile(source/'project.godot',dst/'project.godot')
    p=dst/'scripts/mea_supplied_session.gd'
    s=p.read_text()
    assert s.count('world.thirst = 95.0')==s.count('const SCENE_NAME := "supplied_95"')==1
    p.write_text(s.replace('world.thirst = 95.0','world.thirst = 1.6\n\t\tworld.health = 25.0').replace('const SCENE_NAME := "supplied_95"','const SCENE_NAME := "thirst_challenge_25_1p6"'))
