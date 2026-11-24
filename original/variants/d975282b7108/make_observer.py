"""Add read-only access to the exact cutoff snapshot in a separate game copy."""
from pathlib import Path
import shutil
from resources import PROJECT
root=Path(__file__).resolve().parent/'audit-game'
if root.exists():raise FileExistsError(root)
root.mkdir()
for d in ('scripts','configs','godot_tests'):shutil.copytree(PROJECT/d,root/d)
shutil.copy2(PROJECT/'project.godot',root/'project.godot')
p=root/'scripts/m9_world.gd';src=p.read_text()
a='var navigation_steps: int = 1';assert src.count(a)==1
src=src.replace(a,a+'\nvar archive_landmark: Dictionary = {}')
a='\t\tsuper._advance(1)\n\t\t_sync_deadlines()';assert src.count(a)==1
src=src.replace(a,a+'\n\t\tif tick == 60:\n\t\t\tarchive_landmark = observation().duplicate(true)')
p.write_text(src)
p=root/'scripts/mea_supplied_session.gd';src=p.read_text();a='func handle(request: Dictionary) -> Dictionary:\n';assert src.count(a)==1
p.write_text(src.replace(a,a+'\tif request.get("op", "") == "landmark":\n\t\treturn {"ok": true, "id": request.id, "landmark": world.archive_landmark}\n'))
