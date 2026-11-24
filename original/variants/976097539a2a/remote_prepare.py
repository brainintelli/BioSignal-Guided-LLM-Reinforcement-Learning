import hashlib,importlib,json,os,subprocess,sys,zipfile
from pathlib import Path
r=Path(__file__).resolve().parent;project=r/'project'
source=json.loads((r/'project_source_manifest.json').read_text())
if sys.argv[1:]==['--verify-existing']:
    if not project.is_dir():raise FileNotFoundError(project)
elif sys.argv[1:]:raise ValueError('Unknown operation')
else:
    project.mkdir(exist_ok=False)
    with zipfile.ZipFile(r/'project_frozen.zip') as z:
        if set(z.namelist())!=set(source):raise ValueError('Source inventory')
        for name in source:
            if not (project/name).resolve().is_relative_to(project.resolve()):raise ValueError('Unsafe path')
        z.extractall(project)
for name,h in source.items():
    if hashlib.sha256((project/name).read_bytes()).hexdigest()!=h:raise ValueError('Source '+name)
sys.path.insert(0,str(project));importlib.invalidate_caches()
from agent.mea_shuffle import PIN,StimRecording
if hashlib.sha256((r/'data/manifest.json').read_bytes()).hexdigest()!=PIN:raise ValueError('Data pin')
for phase in ('before','after'):StimRecording(r/'data',pin=PIN,phase=phase)
for path in json.loads((r/'model_config.json').read_text()).values():
    if not Path(path).exists():raise FileNotFoundError(path)
print(json.dumps({'source_files_verified':len(source),'dataset_verified':True,'model_paths_exist':True}),flush=True)
