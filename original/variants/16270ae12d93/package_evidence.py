"""Package immutable comparison evidence after all registered cells finish."""
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(os.environ['SR_ROOT']);RUNTIME=Path(os.environ['SR_RUNTIME']);EXP=ROOT/'experiment01'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
assert (ROOT/'experiment01_driver.exit').read_text().strip()=='0'
assert read(EXP/'driver_status.json')['passed'] and read(EXP/'driver_status.json')['finished_cells']==40
assert read(EXP/'analysis.json')['status']=='complete' and read(EXP/'independent_analysis.json')['passed']
assert read(EXP/'independent_analysis.json')['collection_decisions']<=20480
exits=list((EXP/'logs').glob('*.exit'));assert len(exits)==80 and all(p.read_text().strip()=='0' for p in exits)
progress=[json.loads(s) for s in (EXP/'progress.jsonl').read_text().splitlines()]
owned={e['pid'] for e in progress if e['event']=='stage_started'}
alive=[pid for pid in owned if Path(f'/proc/{pid}').exists()]
ancestors={os.getpid()};parent=os.getppid()
while parent and parent not in ancestors:
    ancestors.add(parent)
    try:parent=int(next(s.split()[1] for s in Path(f'/proc/{parent}/status').read_text().splitlines() if s.startswith('PPid:')))
    except (FileNotFoundError,ProcessLookupError):break
remaining=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdecimal() or int(proc.name) in ancestors:continue
    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    if str(ROOT)+'/' in cmd:remaining.append(dict(pid=int(proc.name),command=cmd))
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
assert not alive and not remaining and not compute.strip(),(alive,remaining,compute)
write(ROOT/'post_run_inventory.json',dict(passed=True,tracked_child_pids=sorted(owned),owned_pids_alive=alive,remaining_task_processes=remaining,excluded_packaging_ancestors=sorted(ancestors),compute=compute,time=time.time()))
manifest_path=RUNTIME.parent/'runtime_manifest.json'
assert sha(manifest_path)=='3f71b79f474aa61d4fa7ebd35d0b07f90b2988ddc01e110aeddb8ca7fed82737'
runtime=read(manifest_path);total=0
for name,entry in runtime['files'].items():
    p=(RUNTIME/name).resolve();assert p.is_relative_to(RUNTIME.resolve()) and p.stat().st_size==entry['size'] and sha(p)==entry['sha256'],name
    total+=entry['size']
write(ROOT/'runtime_audit.json',dict(passed=True,files=len(runtime['files']),bytes=total,manifest_sha256=sha(manifest_path),time=time.time()))
names=[p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.jsonl','.log','.exit','.zip'} and p.name not in {'package.log','package.exit','transport.json','evidence_manifest.json'}]
for folder in ('pilot01','experiment01'):
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file())
assert len(names)==len(set(names)) and not any('__pycache__' in n for n in names)
write(ROOT/'evidence_manifest.json',{n:dict(sha256=sha(ROOT/n),size=(ROOT/n).stat().st_size) for n in sorted(names)})
with tarfile.open(ROOT/'evidence.tar','x') as tar:
    for name in sorted(names+['evidence_manifest.json']):tar.add(ROOT/name,arcname=name,recursive=False)
digest=sha(ROOT/'evidence.tar');write(ROOT/'transport.json',dict(sha256=digest,size=(ROOT/'evidence.tar').stat().st_size))
print(json.dumps(dict(files=len(names)+1,bytes=(ROOT/'evidence.tar').stat().st_size)),flush=True);print(digest[:32]);print(digest[32:])
