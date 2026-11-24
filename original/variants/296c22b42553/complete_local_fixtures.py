"""Supply existing test-only historical fixtures to the isolated project."""
from pathlib import Path
import sys,shutil
r=Path(__file__).resolve().parent;workspace=r.parents[1];project=r/'project'
sys.path.insert(0,str(project))
from agent.m8x_identity import CORE_FILES
added=[]
for name in CORE_FILES:
    dest=project/name
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(workspace/name,dest);added.append(name)
(project/'output/runtime_tests').mkdir(exist_ok=True,parents=True)
print({'existing_test_dependencies_copied':added})
